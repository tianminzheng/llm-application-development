package com.tianyalan.prompt;

import dev.langchain4j.model.input.structured.StructuredPrompt;

@StructuredPrompt({
        "你是一个问题重写者，将输入的问题转换成一个更好的版本，这个版本是为了优化网络搜索。 \n",
        "查看输入并尝试推理其背后的语义意图/含义。 \n",

        "以下是初始问题： \n\n {{question}}. \n\n",
        "改进后的问题，无需前言： \n "
})
public class RewriteQuery {

    private String question;

    public RewriteQuery(String question) {
        this.question = question;
    }
}
