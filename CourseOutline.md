**Course: Working with LLaMA**

**Chapter 1: Open Source LLMs**

**1.1 - ChatGPT vs other Open-Source models**
- The learner will be able to differentiate between closed models like ChatGPT and Open-source models.
- The learner will be able to distinguish between fully open-source models vs open weight models.

**1.2 - Advantages and limitations of Open-Source models**

- The The learner will understand the advantages of open-source models over proprietary models such as flexibility, security and data safety.
- The learner will also comprehend the limitations of the open-source models such as hardware limitations and running costs.

**1.3 – Introduction to LLaMA 2**

- The learner will be able to outline the overall architecture of the LLaMA 2 model from Meta.
- The learner will be able to describe the difference among the different sizes of LLaMA2 model (7B, 13B, 70B) and their pros and cons.
- The learner will be able to compare the training techniques and differences between ChatGPT and LLaMA2.
- The learner will be able to understand the different metrics and scoring techniques of LLM models performance.

**Chapter 2: Working with LLaMA2 model Locally**

**2.1 – Accessing the LLaMA2 Model from Meta**

- The learner will be able to understand how to request permission for access to Llama 2 model from meta in Hugging Face.
- The learner will be able to recognize the responsible use policy of LLaMA2 from meta.
- The learner will be able to differentiate between Llama 2 , Code Llama and Purple Llama model.

**2.2 – Introduction to the llama-cpp-python Library**

- The learner will be able to define what is llama-cpp-python library and explain overall usage and advantages of this library.
- The learner will be able to tell the difference between llama-cpp-python and the LlamaCpp library inside langchain
- The learner will be able to understand the installation process of the GPU and CPU versions of llama-cpp-python.

**2.3 – Understanding GGUF/GPTQ Models and Quantization**

- The learner will be able to describe the difference between GGUF and GPTQ models and how to use them.
- The learner will be able to distinguish between different level of quantization and the tradeoffs between performance and accuracy.

**2.4 – Interacting with the LLaMA2 model with LlamaCpp**

- The learner will be able to load the LLaMA model using the LlamaCpp library and ask questions. 
- The learner will be able to understand different parameters that can be controlled in LlamaCpp such as temperature, max\_tokens and top\_p.
- The learner will be able to define the chat prompt structure of Llama 2

**Chapter 3: Fine Tuning LLaMA2 Model**

**3.1 – Instruction fine-tuning**

- The learner will be able to understand the concept of zero shot and few short learning and their advantages and limitations.
- The learner will be able to comprehend what it means to fully fine-tune an LLM with task-specific examples.
- The learner will be able to explain the concept of catastrophic forgetting in the context of LLM and how to avoid it.

**3.2 – Multi-task Instruction Fine-Tuning**

- The learner will be able to recognize the advantages and limitations of multi-task instruction fine-tuning.
- The learner will be able to observe how a fine-tuned model performs better than a generic model on tasks that it was fine-tuned for.
- The learner will be able to define some of the fundamental LLM evaluation metrics such ROUGE and BLEU scores.
- The learner will learn about different benchmarks such as GLUE, SuperGLUE, MMLU and HELM.

**3.3 – Parameter Efficient Fine-Tuning (PEFT)**

- The learner will be able to explain why fine-tuning a full model layers and weights are very difficult due to HW demand.
- The learner will be able to describe PEFT method and its benefits.
- The learner will be able to distinguish between the different types of PEFT, such as selective, Reparameterization (LoRA) and Additive (Adapters and Soft prompts)

**3.4 – LoRA (Low Rank Adaptation)**

- The learner will be able to explain what Low-rank Adaptation (LoRA) is and its role in parameter-efficient fine-tuning.
- The learner will understand the mechanism of LoRA fine tuning and the concept of decomposition matrices.
- The learner will be able to comprehend the process of training smaller matrices while keeping the original weights of the LLM frozen, and how these matrices are used during inference.
- The learner will recognize the benefits of LoRA, including significant reduction in the number of trainable parameters, enabling fine-tuning with a single GPU.
- The learner will learn how to use the peft library to perform LoRA finetuning on an LLM model.
- The learner will appreciate the performance of LoRA fine-tuned models in comparison to fully fine-tuned models and understand the ongoing research in optimizing the choice of rank for the LoRA matrices.

**3.5 – QLoRA (Quantized Low Rank Adaptation)**

- The learner will be able to distinguish between LoRA and QLoRA techniques for fine-tuning.
- The learner will be able to define 4-bit normal Float, Double Dequantization and will understand the process of quantization.
- The learner will be able to implement QLoRA using bitsandbytes and peft library.
- The learner will be able to compare the different fine-tuning techniques, LoRA, QLoRA, and full fine-tuning.

**Chapter 4: Creating AI Apps Using LangChain** 

**4.1 – Introduction to LangChain**

- The learner will be able to understand the purpose and functionality of LangChain.
- The learner will be able to describe the core components of LangChain.

**4.2 – Prompt Templates and Chaining**

- The learner will be able to understand prompt templates in langchain and will be able to create them.
- The learner will be able to differentiate between Generic and Utility Chains in LangChain.

**4.3 – Managing chat model memory**

- The learner will be able to comprehend the concept of chatbots having no memory association by default. 
- The learner will be able to implement a memory buffer using LangChain to help the bot remember the context of the conversation.
- The learner will be able to implement a summary memory method to shorten longer conversations to save context window size.

**4.4 – Building a Document Q&A and Summarizing App Using LangChain**

- The learner will be able to understand the concept of RAG (Retrieval Augmented Generation) to integrate external documents in the app.
- The learner will be able to explain the concept of vector store database and how to use them in building a RAG storage and Retrieval method.
- The learner will be able to build an app that takes in a pdf document, embeds the data in a vector store database (chromadb or FAISS) and can answer questions about the knowledge inside that document. 
