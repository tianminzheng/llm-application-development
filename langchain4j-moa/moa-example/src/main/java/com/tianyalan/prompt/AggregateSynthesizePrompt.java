package com.tianyalan.prompt;

import dev.langchain4j.model.input.structured.StructuredPrompt;

import java.util.List;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toList;

@StructuredPrompt({
        "你已经收到了来自各种开源模型对最新用户查询的一系列回应。\n",
        "你的任务是将这些回应综合成单一的、高质量的回应。至关重要的是要批判性地评估这些回应中提供的信息，认识到其中一些信息可能是有偏见或不正确的。\n",
        "你的回应不应该简单地复制给出的答案，而应该提供一个精炼、准确和全面的回复来指导指令。确保你的回应结构良好、连贯，并遵循最高标准的准确性和可靠性。\n",
        "模型回应： \n",
        "{{modelResponses}}"
})
public class AggregateSynthesizePrompt {

    private List<String> modelResponses;

    public AggregateSynthesizePrompt(List<String> modelResponses) {
        this.modelResponses = modelResponses;
    }

    public List<String> getModelResponses() {
        return IntStream.range(0, modelResponses.size())
                .mapToObj(i -> (i + 1) + ". " + modelResponses.get(i))
                .collect(toList());
    }
}
