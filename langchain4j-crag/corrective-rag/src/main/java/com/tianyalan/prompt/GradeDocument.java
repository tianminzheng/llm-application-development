package com.tianyalan.prompt;

import dev.langchain4j.model.input.structured.StructuredPrompt;

@StructuredPrompt({
        "你是评估检索文档与用户问题相关性的评分者。\n",

        "这是检索到的文档: \n",

        "{{document}} \n",

        "这是用户的问题: \n",

        "{{question}} \n",


        "如果文档包含与用户问题相关的关键词，则将其评为相关。",
        "这不需要是一个严格的测试。目标是筛选出错误的检索结果。",
        "给出一个二元分数“是”或“否”，以指示文档是否与问题相关。",

        "以JSON格式提供二元分数，使用单一键“score”，无需前缀或解释。"
})
public class GradeDocument {

    private String document;
    private String question;

    public GradeDocument(String document, String question) {
        this.document = document;
        this.question = question;
    }
}
