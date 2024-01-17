# Overview: https://platform.openai.com/docs/models/overview
#
# The OpenAI API is powered by a diverse set of models with different capabilities and price points. 
#
# gpt-4-0613:         Snapshot of gpt-4 from June 13th 2023 with improved function calling support. 
# gpt-3.5-turbo-0613: Snapshot of gpt-3.5-turbo from June 13th 2023. Will be deprecated on June 13, 2024

OPENAI_MODELS = [
    "gpt-4-1106-preview", 
    #The latest GPT-4 model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Returns a maximum of 4,096 output tokens. This preview model is not yet suited for production traffic. 

    "gpt-4-vision-preview", 
    # Ability to understand images, in addition to all other GPT-4 Turbo capabilties. Returns a maximum of 4,096 output tokens. This is a preview model version and not suited yet for production traffic

    "gpt-4",
    # Currently points to gpt-4-0613

    "gpt-4-32k",
    # Currently points to gpt-4-32k-0613
    
    "gpt-3.5-turbo-1106", 
    # The latest GPT-3.5 Turbo model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Returns a maximum of 4,096 output tokens.

    "gpt-3.5-turbo",
    # Currently points to gpt-3.5-turbo-0613.

    "gtp-3.5-instruct",
    # Similar capabilities as GPT-3 era models. Compatible with legacy Completions endpoint and not Chat Completions.

    "babbage-002",
    # Replacement for the GPT-3 ada and babbage base models.

    "davinci-002", 
    # Replacement for the GPT-3 curie and davinci base models.
]