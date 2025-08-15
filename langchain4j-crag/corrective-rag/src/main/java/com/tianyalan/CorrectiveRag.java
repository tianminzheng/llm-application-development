package com.tianyalan;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;

import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;

public interface CorrectiveRag {

    default String answer(String question){
        ensureNotNull(question, "提问");
        return answer(new UserMessage(question)).text();
    }

    AiMessage answer(UserMessage question);
}
