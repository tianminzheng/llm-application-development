package com.tianyalan.internal;

import com.tianyalan.CorrectiveRag;
import com.tianyalan.workflow.CorrectiveNodeFunctions;
import com.tianyalan.workflow.CorrectiveStatefulBean;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.bge.small.en.v15.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.workflow.DefaultStateWorkflow;
import dev.langchain4j.workflow.WorkflowStateName;
import dev.langchain4j.workflow.node.Conditional;
import dev.langchain4j.workflow.node.Node;
import lombok.Builder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import static dev.langchain4j.internal.Utils.getOrDefault;
import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;

public class DefaultCorrectiveRag implements CorrectiveRag {

    private static final Logger log = LoggerFactory.getLogger(DefaultCorrectiveRag.class);

    private final EmbeddingStoreContentRetriever embeddingStoreContentRetriever;
    private final WebSearchContentRetriever webSearchContentRetriever;
    private final ChatLanguageModel chatLanguageModel;
    private final Boolean stream;
    private final Boolean generateWorkflowImage;
    private final Path workflowImageOutputPath;

    @Builder
    public DefaultCorrectiveRag(EmbeddingStoreContentRetriever embeddingStoreContentRetriever,
                                WebSearchContentRetriever webSearchContentRetriever,
                                ChatLanguageModel chatLanguageModel,
                                List<Document> documents,
                                Boolean stream,
                                Boolean generateWorkflowImage,
                                Path workflowImageOutputPath
                                ) {
        if (documents.isEmpty() && embeddingStoreContentRetriever == null) {
            throw new IllegalArgumentException("documents or embeddingStoreContentRetriever must be provided");
        }
        this.embeddingStoreContentRetriever = ensureNotNull(
                getOrDefault(embeddingStoreContentRetriever, DefaultCorrectiveRag.defaultContentRetriever(documents)),
                "embeddingStoreContentRetriever"
        );
        this.webSearchContentRetriever = ensureNotNull(webSearchContentRetriever, "webSearchContentRetriever");
        this.chatLanguageModel = ensureNotNull(chatLanguageModel, "chatLanguageModel");
        this.stream = getOrDefault(stream, false);

        // Check if workflowOutputPath is valid
        if (workflowImageOutputPath != null) {
            this.workflowImageOutputPath = workflowImageOutputPath;
            this.generateWorkflowImage = true;
        } else {
            this.workflowImageOutputPath = null;
            this.generateWorkflowImage = getOrDefault(generateWorkflowImage, false);
        }

    }

    @Override
    public AiMessage answer(UserMessage question) {
        // 定义有状态对象
        CorrectiveStatefulBean statefulBean = new CorrectiveStatefulBean();
        statefulBean.setQuestion(question.singleText());

        // 创建工作流
        DefaultStateWorkflow<CorrectiveStatefulBean> wf = correctiveWorkflow(statefulBean);

        // 执行工作流
        wf.run();

        // 打印状态转换过程
        log.debug("转换过程: \n" + wf.prettyTransitions() + "\n");

        // 打印最终答案
        String finalAnswer = statefulBean.getGeneration();
        log.info("最终答案: \n" + finalAnswer);

        // 生成工作流图片
        if (generateWorkflowImage) {
            try {
                generateWorkflowImage(wf);
            } catch (Exception e) {
                log.warn("生成工作流图片失败", e);
            }
        }
        return AiMessage.from(finalAnswer);
    }

    private DefaultStateWorkflow<CorrectiveStatefulBean> correctiveWorkflow(CorrectiveStatefulBean statefulBean) {
        // 创建CorrectiveNodeFunctions
        CorrectiveNodeFunctions cwf = new CorrectiveNodeFunctions.Builder()
                .withEmbeddingStoreContentRetriever(embeddingStoreContentRetriever)
                .withChatLanguageModel(chatLanguageModel)
                .withWebSearchContentRetriever(webSearchContentRetriever)
                .build();

        // 定义Node
        Function<CorrectiveStatefulBean, CorrectiveStatefulBean> retrieve = state -> cwf.retrieve(statefulBean);
        Function<CorrectiveStatefulBean, CorrectiveStatefulBean> generate = state -> cwf.generate(statefulBean);
        Function<CorrectiveStatefulBean, CorrectiveStatefulBean> gradeDocuments = state -> cwf.gradeDocuments(statefulBean);
        Function<CorrectiveStatefulBean, CorrectiveStatefulBean> rewriteQuery = state -> cwf.transformQuery(statefulBean);
        Function<CorrectiveStatefulBean, CorrectiveStatefulBean> webSearch = state -> cwf.webSearch(statefulBean);

        // 创建Node
        Node<CorrectiveStatefulBean, CorrectiveStatefulBean> retrieveNode = Node.from("Retrieve Node", retrieve);
        Node<CorrectiveStatefulBean, CorrectiveStatefulBean> generateNode = Node.from("Generate Node", generate);
        Node<CorrectiveStatefulBean, CorrectiveStatefulBean> gradeDocumentsNode = Node.from("Grade Node", gradeDocuments);
        Node<CorrectiveStatefulBean, CorrectiveStatefulBean> rewriteQueryNode = Node.from("Re-Write Query Node", rewriteQuery);
        Node<CorrectiveStatefulBean, CorrectiveStatefulBean> webSearchNode = Node.from("WebSearch Node", webSearch);

        // 构建工作流图
        DefaultStateWorkflow<CorrectiveStatefulBean> wf = DefaultStateWorkflow.<CorrectiveStatefulBean>builder()
                .statefulBean(statefulBean)
                .addNodes(Arrays.asList(retrieveNode, generateNode, gradeDocumentsNode, rewriteQueryNode, webSearchNode))
                .build();

        //  定义Edge
        wf.putEdge(retrieveNode, gradeDocumentsNode); // retrieveNode -> gradeDocumentsNode
        wf.putEdge(gradeDocumentsNode, Conditional.eval(obj -> { // gradeDocumentsNode -> rewriteQueryNode OR generateNode
            if (obj.getWebSearch().equals("是")) {
                log.info("---决策: 所有文档都与问题无关，转换查询---");
                return rewriteQueryNode;
            }else {
                log.info("---决策: 生成---");
                return generateNode;
            }
        }));
        wf.putEdge(rewriteQueryNode, webSearchNode);
        wf.putEdge(webSearchNode, generateNode);
        wf.putEdge(generateNode, WorkflowStateName.END);
        // Define node entrypoint
        wf.startNode(retrieveNode);
        return wf;
    }

    private static EmbeddingStoreContentRetriever defaultContentRetriever(List<Document> documents) {
        EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .documentSplitter(DocumentSplitters.recursive(250, 0))
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();

        ingestor.ingest(documents);
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .minScore(0.6)
                .build();
    }

    private void generateWorkflowImage(DefaultStateWorkflow<CorrectiveStatefulBean> wf) throws IOException {
        if (workflowImageOutputPath != null) {
            wf.generateWorkflowImage(workflowImageOutputPath.toAbsolutePath().toString());
        } else {
            wf.generateWorkflowImage();
        }
    }

}
