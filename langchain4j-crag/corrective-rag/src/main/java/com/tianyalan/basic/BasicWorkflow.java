package com.tianyalan.basic;

import dev.langchain4j.workflow.DefaultStateWorkflow;
import dev.langchain4j.workflow.WorkflowStateName;
import dev.langchain4j.workflow.node.Conditional;
import dev.langchain4j.workflow.node.Node;

import java.io.IOException;
import java.util.Arrays;
import java.util.function.Function;

public class BasicWorkflow {
        public static void main(String[] args) throws IOException {

            // 定义一个有状态Bean
            class MyStatefulBean {
                int value = 0;
            }
            MyStatefulBean myStatefulBean = new MyStatefulBean();

            // 为每个节点定义函数
            Function<MyStatefulBean, String> node1Func = obj -> {
                obj.value +=1;
                System.out.println("节点1: [" + obj.value + "]");
                return "节点1: 函数被执行";
            };
            Function<MyStatefulBean, String> node2Func = obj -> {
                obj.value +=2;
                System.out.println("节点2: [" + obj.value + "]");
                return "节点2: 函数被执行";
            };
            Function<MyStatefulBean, String> node3Func = obj -> {
                obj.value +=3;
                System.out.println("节点3: [" + obj.value + "]");
                return "节点3: 函数被执行";
            };
            Function<MyStatefulBean, String> node4Func = obj -> {
                obj.value +=4;
                System.out.println("节点4: [" + obj.value + "]");
                return "节点4: 函数被执行";
            };

            // 创建节点
            Node<MyStatefulBean, String> node1 = Node.from("node1", node1Func);
            Node<MyStatefulBean, String> node2 = Node.from("node2", node2Func);
            Node<MyStatefulBean, String> node3 = Node.from("node3", node3Func);
            Node<MyStatefulBean, String> node4 = Node.from("node4", node4Func);

            // 创建工作流
            DefaultStateWorkflow<MyStatefulBean> workflow = DefaultStateWorkflow.<MyStatefulBean>builder() //DefaultWorkflowTmp.addStatefulBan(myStatefulBean).build();
                    .statefulBean(myStatefulBean)
                    .addNodes(Arrays.asList(node1, node2, node3))
                    .build();

            // 添加节点到工作流
            workflow.addNode(node1);
            workflow.addNode(node2);
            workflow.addNode(node3);
            workflow.addNode(node4);

            // 定义边
            workflow.putEdge(node1, node2);
            workflow.putEdge(node2, node3);
            workflow.putEdge(node3, Conditional.eval(obj -> {
                System.out.println("状态值 [" + obj.value + "]");
                if (obj.value > 6) {
                    return node4;
                } else {
                    return node2;
                }
            }));

            workflow.putEdge(node4, WorkflowStateName.END);

            // 启动工作流
            workflow.startNode(node1);

            // 执行工作流
            workflow.run();

            // 打印转换过程
            String transitions = workflow.prettyTransitions();
            System.out.println("转换过程: \n");
            System.out.println(transitions);

            // Generate workflow image
            workflow.generateWorkflowImage("workflow.svg");
        }
}
