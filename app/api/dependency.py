from fastapi import Depends

from app.service.follow_up_care import FollowupCareService
from app.service.medical_inquiry import MedicalInquiryService

async def get_medical_inquiry_service() -> MedicalInquiryService:
    return MedicalInquiryService()
    
async def get_follow_up_care_service() -> FollowupCareService:
    return FollowupCareService()
