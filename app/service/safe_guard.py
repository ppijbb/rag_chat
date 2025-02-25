import json

from typing import Any, List
import time
import traceback

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse, Response

from ray import serve

from transformers import AutoTokenizer

from app.core.langchain_module.guard import LMTextClassifier, GuardVotingClassifier, DiserializedOVModelForSequenceClassification
from app.service._base import BaseService

# @serve.deployment
# @serve.ingress(app=router)
class GaurdService(BaseService):
    label_classes: List[str] = ["BENIGN", "INJECTION", "JAILBREAK"]
    device:str = "cpu"
    model_name:str = "katanemolabs/Arch-Guard-cpu"
    model:GuardVotingClassifier = GuardVotingClassifier(
            estimators=[
                # ('prompt guard1', LMTextClassifier(
                #     model="meta-llama/Prompt-Guard-86M",
                #     device=device,
                #     label_classes=label_classes)),
                # ('prompt guard2', LMTextClassifier(
                #     model="katanemo/Arch-Guard",
                #     device=device,
                #     label_classes=label_classes)),
                # ('prompt guard3', LMTextClassifier(
                #     model="Niansuh/Prompt-Guard-86M",
                #     device=device,
                #     label_classes=label_classes))
                ('prompt guard1', LMTextClassifier(
                    model=DiserializedOVModelForSequenceClassification.from_pretrained( # pg
                        model_id="meta-llama/Prompt-Guard-86M", 
                        device_map=device,
                        low_cpu_mem_usage=True,
                        load_in_4bit=True,
                        trust_remote_code=True),
                    tokenizer=AutoTokenizer.from_pretrained(
                        pretrained_model_name_or_path="meta-llama/Prompt-Guard-86M",
                        trust_remote_code=True),
                    device=device,
                    label_classes=label_classes))
                ('arch guard', LMTextClassifier(
                    model=DiserializedOVModelForSequenceClassification.from_pretrained( # ag
                        model_id=model_name,
                        device_map=device,
                        low_cpu_mem_usage=True,
                        load_in_4bit=True,
                        trust_remote_code=True),
                    tokenizer=AutoTokenizer.from_pretrained(
                        pretrained_model_name_or_path=model_name,
                        trust_remote_code=True),
                    device=device,
                    label_classes=label_classes))
                ],
                voting='soft'
            ).fit(X=label_classes, y=label_classes)
    
    
    def get_prompt_guards(self, *args, **kwargs):
        return [
            ('prompt guard1', LMTextClassifier(
                model="meta-llama/Prompt-Guard-86M",
                device='cpu',
                label_classes=self.label_classes)),
            # ('prompt guard2', LMTextClassifier(
            #     model="katanemo/Arch-Guard",
            #     device='cpu',
            #     label_classes=self.label_classes)),
            # ('prompt guard3', LMTextClassifier(
            #     model="Niansuh/Prompt-Guard-86M",
            #     device='cpu',
            #     label_classes=self.label_classes))
            ]

    def get_prompt_clf(self, *args, **kwargs):
        return ('arch guard', LMTextClassifier(
                    model=DiserializedOVModelForSequenceClassification.from_pretrained(
                        model_id=self.model_name, 
                        device_map=self.device,
                        low_cpu_mem_usage=True,
                        load_in_4bit=True,
                        trust_remote_code=True
                    ),
                    tokenizer=AutoTokenizer.from_pretrained(
                        pretrained_model_name_or_path=self.model_name,
                        trust_remote_code=True
                    ),
                    device='cpu',
                    label_classes=self.label_classes))
        
    def get_classifier(self, *args, **kwargs):
        return GuardVotingClassifier(
            estimators=self.get_prompt_guards()+[self.get_prompt_clf()],
            voting='soft'
        ).fit(X=self.label_classes, y=self.label_classes)

    @serve.batch(
        max_batch_size=4, 
        batch_wait_timeout_s=0.1)
    async def batched_process(
       self,
       request_text: List[Any],
       *args,
       **kwargs
    ) -> List[str]:
        self_class = self[0]._get_class() # ray batch wrapper 에서 self가 list로 들어옴
        self_class.server_logger.info(f"Batched request: {len(request_text)}")
        return await self_class.model.predict.remote(X=request_text)
        
    def get_service_chain(self, *args, **kwargs):
        return self.model
