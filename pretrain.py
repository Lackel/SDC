from utils.utils import *
from model import *
from dataloader import *
from model import BertForModel
from transformers import WEIGHTS_NAME, CONFIG_NAME, logging
import warnings
from init_parameter import init_model

class PretrainModelManager:
    def __init__(self, args, data):
        set_seed(args.seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = BertForModel(args.bert_model, num_labels=data.n_known_cls)
        self.model.to(self.device)

        self.num_train_optimization_steps = int(
            len(data.train_labeled_examples) / args.train_batch_size) * args.num_pretrain_epochs

        self.optimizer, self.scheduler = self.get_optimizer(args)
        self.best_eval_score = 0

    def eval(self, args, data):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.n_known_cls)).to(self.device)

        for batch in tqdm(data.eval_dataloader, desc="pre-training-eval"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.set_grad_enabled(False):
                _, logits = self.model(X)
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)

        return acc 

    def train(self, args, data):

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        wait = 0
        best_model = None
        mlm_iter = iter(data.train_semi_dataloader)

        for epoch in range(int(args.num_pretrain_epochs)):
            print('---------------------------')
            print(f'pre-training epoch:{epoch}')
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(data.train_labeled_dataloader, desc="Pre-training")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
                try:
                    batch = mlm_iter.next()
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids, labels = batch
                except StopIteration:
                    mlm_iter = iter(data.train_semi_dataloader)
                    batch = mlm_iter.next()
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids, labels = batch
                X_mlm = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}

                mask_ids, mask_lb = mask_tokens(X_mlm['input_ids'].cpu(), tokenizer)
                X_mlm["input_ids"] = mask_ids.to(self.device)

                with torch.set_grad_enabled(True):
                    features, logits = self.model(X)
                    features = F.normalize(features, dim=1)
                    if isinstance(self.model, nn.DataParallel):
                        loss_src = self.model.module.loss_ce(logits, label_ids)
                        loss_mlm = self.model.module.mlmForward(X_mlm, mask_lb.to(self.device))
                    else:
                        loss_src = self.model.loss_ce(logits, label_ids)
                        loss_mlm = self.model.mlmForward(X_mlm, mask_lb.to(self.device))
                    lossTOT = loss_src + loss_mlm
                    lossTOT.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    tr_loss += lossTOT.item()

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            eval_score = self.eval(args, data)
            print(f'train_loss:{loss}, eval_score:{eval_score}')

            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.pre_wait_patient:
                    break

        self.model = best_model
        if args.save_model:
            self.save_model(args)

    def get_optimizer(self, args):
        num_warmup_steps = int(args.warmup_proportion * self.num_train_optimization_steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_pre)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=self.num_train_optimization_steps)
        return optimizer, scheduler

    def get_features_labels(self, dataloader, model, args):
        model.eval()
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation for clustering"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                feature, _ = model(X, output_hidden_states=True)

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def evaluation(self, args, data):

        feats, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats = feats.cpu().numpy()

        km = KMeans(n_clusters=data.num_labels, n_init=20).fit(feats)

        y_pred = km.labels_
        y_true = labels.cpu().numpy()

        results = clustering_score(y_true, y_pred, data.known_lab)
        print('results', results)

    def load_model(self, args):
        model_file = os.path.join(args.dataset, args.pretrain_dir + '_' + str(args.seed), 'premodel.pth')
        self.model.load_state_dict(torch.load(model_file))

    def save_model(self, args):
        file_dir = os.path.join(args.dataset, args.pretrain_dir + '_' + str(args.seed))
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        model_file = os.path.join(args.dataset, args.pretrain_dir + '_' + str(args.seed), 'premodel.pth')
        torch.save(self.model.state_dict(), model_file)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    data = Data(args)

    print('Pre-training begin...')
    manager_pre = PretrainModelManager(args, data)
    manager_pre.train(args, data)
    print('Pre-training finished!')