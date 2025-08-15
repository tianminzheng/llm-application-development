package com.tianyalan.internal;

import com.tianyalan.AgentChatLanguageModel;
import com.tianyalan.AggregatorChatLanguageModel;
import com.tianyalan.AggregatorStreamingChatLanguageModel;
import com.tianyalan.MixtureOfAgents;
import com.tianyalan.workflow.MoaNodeFunctions;
import com.tianyalan.workflow.MoaStatefulBean;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.workflow.DefaultStateWorkflow;
import dev.langchain4j.workflow.WorkflowStateName;
import dev.langchain4j.workflow.node.Node;
import lombok.Builder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.stream.IntStream;

import static dev.langchain4j.internal.Utils.getOrDefault;
import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;
import static java.util.stream.Collectors.toList;

public class DefaultMixtureOfAgents implements MixtureOfAgents {

    private static final Logger log = LoggerFactory.getLogger(DefaultMixtureOfAgents.class);
    private final List<AgentChatLanguageModel> refLlms;
    private final Integer numberOfLayers;
    private final Optional<AggregatorStreamingChatLanguageModel> generateStreamingLlm;
    private final Optional<AggregatorChatLanguageModel> generateLlm;
    private final Boolean stream;
    private final Boolean generateWorkflowImage;
    private final Path workflowImageOutputPath;
    private Map<Integer, List<Node<MoaStatefulBean,MoaStatefulBean>>> layers;

    @Builder
    public DefaultMixtureOfAgents(List<AgentChatLanguageModel> refLlms,
                                  Integer numberOfLayers, 
                                  AggregatorStreamingChatLanguageModel generateStreamingLlm,
                                  AggregatorChatLanguageModel generateLlm,
                                  Boolean stream, 
                                  Boolean generateWorkflowImage, 
                                  Path workflowImageOutputPath) {
        this.generateStreamingLlm = Optional.ofNullable(generateStreamingLlm);
        this.generateLlm = Optional.ofNullable(generateLlm);
        if (generateStreamingLlm == null && generateLlm == null) {
            throw new IllegalArgumentException("Either generateLlm or generateStreamingLlm must be provided");
        }
        if (generateStreamingLlm != null && generateLlm != null) {
            throw new IllegalArgumentException("Both generateLlm and generateStreamingLlm cannot be provided");
        }

        this.refLlms = ensureNotNull(
                getOrDefault(refLlms, DefaultMixtureOfAgents::defaultRefLlms),
                "refLlms");
        this.numberOfLayers = getOrDefault(numberOfLayers, 1);
        this.stream = getOrDefault(stream, false);

        this.generateWorkflowImage = workflowImageOutputPath != null || Boolean.TRUE.equals(generateWorkflowImage);
        this.workflowImageOutputPath = workflowImageOutputPath;
    }

    @Override
    public AiMessage answer(UserMessage question) {
        MoaStatefulBean statefulBean = processQuestion(question);
        return AiMessage.from(statefulBean.getGeneration());
    }

    @Override
    public List<String> answerStream(UserMessage question) {
        MoaStatefulBean statefulBean = processQuestion(question);
        return statefulBean.getGeneratedStream();
    }

    private MoaStatefulBean processQuestion(UserMessage question) {
        MoaStatefulBean statefulBean = new MoaStatefulBean();
        statefulBean.setQuestion(question.singleText());

        DefaultStateWorkflow<MoaStatefulBean> wf = moaWorkflow(statefulBean);

        if (stream) {
            log.info("Running workflow in stream mode...");
            wf.runStream(node -> log.debug("Processing node: " + node.getName()));
        } else {
            log.info("Running workflow in normal mode...");
            wf.run();
        }

        log.debug("Transitions: \n" + wf.prettyTransitions() + "\n");

        if (generateWorkflowImage) {
            try {
                generateWorkflowImage(wf);
            } catch (Exception e) {
                log.warn("Error generating workflow image", e);
            }
        }
        return statefulBean;
    }

    private DefaultStateWorkflow<MoaStatefulBean> moaWorkflow (MoaStatefulBean statefulBean){
        log.info("=== Generating MOA architecture.. ===");
        layers = new ConcurrentHashMap<>();
        MoaNodeFunctions moaNodeFunctions = new MoaNodeFunctions();

        IntStream.rangeClosed(1, numberOfLayers).forEach(iLayer -> {
            List<Node<MoaStatefulBean, MoaStatefulBean>> nodes = IntStream.rangeClosed(1, refLlms.size())
                    .mapToObj(iLlm -> createAgentNode(iLayer, iLlm, refLlms.get(iLlm - 1), moaNodeFunctions))
                    .collect(toList());
            log.debug("  === Created Layer: [" + iLayer + "], Nodes added [" + nodes.size() + "] ===");
            layers.putIfAbsent(iLayer, nodes);
        });

        Node<MoaStatefulBean, MoaStatefulBean> aggregatorNode = createAggregatorNode(moaNodeFunctions);
        log.debug("  === Created Aggregator Node ===");

        DefaultStateWorkflow<MoaStatefulBean> wf = buildWorkflow(statefulBean, aggregatorNode);

        log.info("=== MOA architecture generated ===");
        log.info("  === Layers: [" + layers.size() + "], Agents: [" + layers.values().stream().mapToInt(List::size).sum() + "] ===");
        log.info("  === Agent Aggregator: [" + aggregatorNode.getName() + "] ===");
        log.info("Parsing MOA architecture to workflow...");
        return wf;
    }

    private Node<MoaStatefulBean, MoaStatefulBean> createAgentNode(int iLayer, int iLlm, AgentChatLanguageModel refLlm, MoaNodeFunctions moaNodeFunctions) {
        Function<MoaStatefulBean, MoaStatefulBean> proposerAgent = obj -> moaNodeFunctions.proposerAgent(obj, refLlm, iLayer);
        return Node.from("AgentNode " + iLayer + "." + iLlm + ": " + refLlm.name(), proposerAgent);
    }

    private Node<MoaStatefulBean, MoaStatefulBean> createAggregatorNode(MoaNodeFunctions moaNodeFunctions) {
        if (generateStreamingLlm.isPresent()) {
            Function<MoaStatefulBean, MoaStatefulBean> aggregator = obj -> moaNodeFunctions.aggregatorStreamingAgent(obj, generateStreamingLlm.get());
            return Node.from("AggregatorStreamingNode: " + generateStreamingLlm.get().name(), aggregator);
        } else {
            Function<MoaStatefulBean, MoaStatefulBean> aggregator = obj -> moaNodeFunctions.aggregatorAgent(obj, generateLlm.get());
            return Node.from("AggregatorNode: " + generateLlm.get().name(), aggregator);
        }
    }

    private DefaultStateWorkflow<MoaStatefulBean> buildWorkflow(MoaStatefulBean statefulBean, Node<MoaStatefulBean, MoaStatefulBean> aggregatorNode) {
        DefaultStateWorkflow<MoaStatefulBean> wf = DefaultStateWorkflow.<MoaStatefulBean>builder()
                .statefulBean(statefulBean)
                .addNodes(layers.values().stream().flatMap(List::stream).collect(toList()))
                .addNode(aggregatorNode)
                .build();

        for (int iLayer = 1; iLayer <= layers.size(); iLayer++) {
            List<Node<MoaStatefulBean, MoaStatefulBean>> nodes = layers.get(iLayer);
            for (int iNode = 0; iNode < nodes.size(); iNode++) {
                Node<MoaStatefulBean, MoaStatefulBean> currentNode = nodes.get(iNode);
                Node<MoaStatefulBean, MoaStatefulBean> nextNode = getNextNode(iLayer, iNode, nodes, aggregatorNode);
                wf.putEdge(currentNode, nextNode);
            }
        }
        wf.putEdge(aggregatorNode, WorkflowStateName.END);

        wf.startNode(layers.get(1).get(0));

        return wf;
    }

    private Node<MoaStatefulBean, MoaStatefulBean> getNextNode(int iLayer, int iNode, List<Node<MoaStatefulBean, MoaStatefulBean>> nodes, Node<MoaStatefulBean, MoaStatefulBean> aggregatorNode) {
        if (iNode < nodes.size() - 1) {
            return nodes.get(iNode + 1);
        } else if (iLayer < layers.size()) {
            return layers.get(iLayer + 1).get(0);
        } else {
            return aggregatorNode;
        }
    }

    private static List<AgentChatLanguageModel> defaultRefLlms(){
        return null;
    }

    private void generateWorkflowImage(DefaultStateWorkflow<MoaStatefulBean> wf) throws IOException {
        if (workflowImageOutputPath != null) {
            wf.generateWorkflowImage(workflowImageOutputPath.toAbsolutePath().toString());
        } else {
            wf.generateWorkflowImage();
        }
    }
}
