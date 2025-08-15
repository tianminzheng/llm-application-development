package com.tianyalan.prompt;

import dev.langchain4j.model.input.structured.StructuredPrompt;

@StructuredPrompt({
        "你是一个问题回答任务的助手",
        "使用以下检索到的上下文片段来回答这个问题。",
        "如果你不知道答案，就直接说你不知道。",
        "最多使用三句话，并保持答案简洁。",

        "问题: {{question}} \n\n",
        "上下文: {{context}} \n\n",
        "答案:"
})
public class GenerateAnswer {

    private String question;
    private String context;

    public GenerateAnswer(String question, String context) {
        this.question = question;
        this.context = context;
    }
}
