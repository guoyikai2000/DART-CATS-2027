"""Binary last-token classifier; preserve the reference backbone and dense LoRA."""

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from data import LABELS


class LastTokenClassifier(PreTrainedModel):
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True
    accepts_loss_kwargs = False

    def __init__(self, backbone: PreTrainedModel):
        super().__init__(backbone.config)
        self.backbone = backbone
        self.config.num_labels = len(LABELS)
        self.config.label2id = LABELS.copy()
        self.config.id2label = {value: key for key, value in LABELS.items()}
        text_config = self.config.get_text_config()
        self.score = nn.Linear(text_config.hidden_size, len(LABELS), bias=False,
                               dtype=torch.float32)
        nn.init.normal_(self.score.weight, std=text_config.initializer_range)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels=None, **_):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False, return_dict=True).last_hidden_state
        last_token = attention_mask.long().sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_token]
        logits = self.score(pooled.float())
        loss = None if labels is None else F.cross_entropy(logits.float(), labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def load_classifier(model_path: str, precision: str = "bf16") -> LastTokenClassifier:
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    backbone = AutoModel.from_pretrained(model_path, dtype=dtype,
                                        attn_implementation="eager", local_files_only=True)
    return LastTokenClassifier(backbone)


def create_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                             fix_mistral_regex=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    model = load_classifier(args.model_name_or_path, args.precision)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.get_text_config().pad_token_id = tokenizer.pad_token_id
    suffixes = tuple(args.lora_targets.split(","))
    targets = [name for name, module in model.named_modules()
               if isinstance(module, nn.Linear) and name.endswith(suffixes)
               and "vision" not in name]
    if not targets:
        raise ValueError("No text-backbone LoRA targets were found")
    config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_rank,
                        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        target_modules=targets, modules_to_save=["score"], bias="none")
    model = get_peft_model(model, config)
    model.resolved_lora_targets = targets
    return model, tokenizer
