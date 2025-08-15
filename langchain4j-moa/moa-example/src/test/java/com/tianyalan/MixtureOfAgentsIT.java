package com.tianyalan;

import dev.langchain4j.model.anthropic.AnthropicStreamingChatModel;
import com.tianyalan.internal.DefaultMixtureOfAgents;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.chat.StreamingChatLanguageModel;
import dev.langchain4j.model.dashscope.QwenChatModel;
import dev.langchain4j.model.dashscope.QwenModelName;
import dev.langchain4j.model.mistralai.MistralAiChatModel;
import dev.langchain4j.model.mistralai.MistralAiChatModelName;

import org.junit.jupiter.api.Test;

import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import static java.lang.System.getenv;
import static org.assertj.core.api.Assertions.assertThat;

class MixtureOfAgentsIT {

    String MISTRAL_API_KEY = "qWbLJB1RvlyvBF3Aso50OEKtK1OSQxca";
    String DASHSCOPE_API_KEY = "sk-615d995a9b654a8587dd257312464238";

    ChatLanguageModel openMixtral822B = MistralAiChatModel.builder()
            .apiKey(MISTRAL_API_KEY)
            .modelName(MistralAiChatModelName.OPEN_MIXTRAL_8X22B)
            .temperature(0.7)
            .build();

    ChatLanguageModel openMixtral87B = MistralAiChatModel.builder()
            .apiKey(MISTRAL_API_KEY)
            .modelName(MistralAiChatModelName.OPEN_MIXTRAL_8x7B)
            .temperature(0.7)
            .build();

    ChatLanguageModel qwen1572B = QwenChatModel.builder()
            .apiKey(DASHSCOPE_API_KEY)
            .modelName(QwenModelName.QWEN1_5_72B_CHAT)
            .build();

    ChatLanguageModel qwen272B = QwenChatModel.builder()
            .apiKey(DASHSCOPE_API_KEY)
            .modelName(QwenModelName.QWEN2_72B_INSTRUCT)
            .build();

    List<AgentChatLanguageModel> refLlms = Arrays.asList(
            AgentChatLanguageModel.from("openMixtral822B", openMixtral822B),
            AgentChatLanguageModel.from("openMixtral87B", openMixtral87B),
            AgentChatLanguageModel.from("qwen1572B", qwen1572B),
            AgentChatLanguageModel.from("qwen272B", qwen272B)
    );

    MixtureOfAgents moa = DefaultMixtureOfAgents.builder()
            .refLlms(refLlms)
            .numberOfLayers(2)
            .generateLlm(AggregatorChatLanguageModel.from("openMixtral822B",openMixtral822B))
            .workflowImageOutputPath(Paths.get("images/moa-wf-4.svg"))
            .build();

    @Test
    void run_using_builder_mandatory_params(){
        String question = "在杭州最值得做的事情有哪些？";
        String answer = moa.answer(question);
        System.out.println(answer);
    }

    @Test
    void run_using_generateStreamingModel() {
        StreamingChatLanguageModel streamingAnthropic = AnthropicStreamingChatModel.builder()
                .apiKey(getenv("ANTHROPIC_API_KEY"))
                .logRequests(true)
                .logResponses(true)
                .build();

        MixtureOfAgents moa = DefaultMixtureOfAgents.builder()
                .refLlms(refLlms)
                .numberOfLayers(2)
                .generateStreamingLlm(
                        AggregatorStreamingChatLanguageModel.from(
                                "streamingAnthropic",
                                streamingAnthropic))
                .build();

        String question = "在杭州最值得做的事情有哪些？";

        // when
        List<String> finalStream = moa.answerStream(question);

        // then
        assertThat(finalStream)
                .anySatisfy(chunk -> {
                    assertThat(chunk).containsIgnoringWhitespaces("西湖");
                });
    }
}
