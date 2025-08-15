package com.tianyalan;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.UrlDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.transformer.HtmlTextExtractor;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import com.tianyalan.internal.DefaultCorrectiveRag;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_O;
import static org.assertj.core.api.Assertions.assertThat;


class CorrectiveRagIT {

    static String apiKey = "请输入你的APIKey";

    // 建立文档索引
    List<Document> documents = loadDocuments(
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
    );

    // 定义ChatLanguageModel
    ChatLanguageModel llm = OpenAiChatModel.builder()
            .apiKey(apiKey)
            .modelName(GPT_4_O)
            .temperature(0.0)
            .build();

    // 定义WebContentRetriever
    WebSearchContentRetriever webRetriever = WebSearchContentRetriever.builder()
            .webSearchEngine(TavilyWebSearchEngine.builder().apiKey("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").build())
            .maxResults(3)
            .build();

    // 创建CorrectiveRag实例
    CorrectiveRag correctiveRag = DefaultCorrectiveRag.builder()
            .documents(documents)
            .webSearchContentRetriever(webRetriever)
            .chatLanguageModel(llm)
            .build();

    @Test
    void run_using_default_embeddingContentRetriever() {
        // given
        String question = "杭州市一座什么样的城市？";

        // when
        String answer = correctiveRag.answer(question);
        System.out.println(answer);

        // then
        assertThat(answer).containsIgnoringWhitespaces("code generation");
    }

    @Test
    void run_using_default_embeddingContentRetriever_and_generate_workflow_image_with_outputPath() {
        // given
        Path workflowImageOutputPath = Paths.get("images/corrective-wf.svg");
        CorrectiveRag correctiveRagWithWorkflowImage = DefaultCorrectiveRag.builder()
                .documents(documents)
                .webSearchContentRetriever(webRetriever)
                .chatLanguageModel(llm)
                .workflowImageOutputPath(workflowImageOutputPath)
                .build();

        String question = "How does the AlphaCodium paper work?";

        // when
        String answer = correctiveRagWithWorkflowImage.answer(question);
        System.out.println(answer);

        // then
        assertThat(answer).containsIgnoringWhitespaces("code generation");
        assertThat(workflowImageOutputPath.toFile()).exists();

    }

    private static List<Document> loadDocuments(String... uris) {
        List<Document> documents = new ArrayList<>();
        for (String uri : uris) {
            Document document = UrlDocumentLoader.load(uri,new TextDocumentParser());
            HtmlTextExtractor transformer = new HtmlTextExtractor(null, null, false);
            document = transformer.transform(document);
            documents.add(document);
        }
        return documents;
    }
}
