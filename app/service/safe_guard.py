import json

from typing import Any, List
import time
import traceback

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse, Response

from ray import serve

from transformers import AutoTokenizer

from app.service.follow_up_care import FollowupCareService
from app.core.langchain_module.guard import LMTextClassifier, CustomVotingClassifier, DiserializedOVModelForSequenceClassification
from app.api.controller.base_router import BaseIngress

# @serve.deployment
# @serve.ingress(app=router)
class GaurdService(BaseIngress):
    routing = False
    prefix = "/medical_inquiry"
    tags = ["Medical Inquiry"]
    include_in_schema = False
   
    def __init__(
        self, 
        service: FollowupCareService = None
    ) -> None:
        super().__init__(service=service)
        label_classes = ["BENIGN", "INJECTION", "JAILBREAK"]
    
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
        
    def get_classifier(self, *args, **kwargs):
        device = "cpu"
        model_name = "katanemolabs/Arch-Guard-cpu"

        return CustomVotingClassifier(
            estimators=self.get_prompt_guards()+[
                ('arch guard', LMTextClassifier(
                    model=DiserializedOVModelForSequenceClassification.from_pretrained(
                        model_id=model_name, 
                        device_map=device,
                        low_cpu_mem_usage=True,
                        load_in_4bit=True,
                        trust_remote_code=True
                    ),
                    tokenizer=AutoTokenizer.from_pretrained(
                        pretrained_model_name_or_path=model_name,
                        trust_remote_code=True
                    ),
                    device='cpu',
                    label_classes=self.label_classes))
                ],
            voting='soft'
        ).fit(X=self.label_classes, y=self.label_classes)

    @serve.batch(
            max_batch_size=4, 
            batch_wait_timeout_s=0.1)
    async def batched_process(
       self,
       request_prompt: List[Any],
       request_text: List[Any]
    ) -> List[str]:
        self_class = self[0]._get_class() # ray batch wrapper 에서 self가 list로 들어옴
        self_class.server_logger.info(f"Batched request: {len(request_text)}")
        return await self_class.service.summarize.remote(
            input_prompt=request_prompt,
            input_text=request_text,
            batch=True)
    
   