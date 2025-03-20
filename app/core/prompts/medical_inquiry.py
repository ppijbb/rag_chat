from app.core.prompts.live_prompt import LivePromptObject

SYSTEM_PROMPT:str = LivePromptObject("medical_inquiry/system_prompt.txt")
STEP2_SYSTEM_PROMPT:str = LivePromptObject("medical_inquiry/step2_system_prompt.txt")
ENTITY_PROMPT_KO:str = LivePromptObject("medical_inquiry/entity_prompt_ko.txt")
ENTITY_PROMPT_EN:str = LivePromptObject("medical_inquiry/entity_prompt_en.txt")
TIMER_PROMPT:str = LivePromptObject("medical_inquiry/timer_prompt.txt")
MULTI_QUERY_PROMPT:str = LivePromptObject("medical_inquiry/multi_query_prompt.txt")
STEP_CONTROL_PROMPT:str = LivePromptObject("medical_inquiry/step_control_prompt.txt")
TREATMENT_PROMPT:str = LivePromptObject("medical_inquiry/treatment_prompt.txt")
