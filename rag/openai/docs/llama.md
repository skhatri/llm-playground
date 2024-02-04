LLaMA: Open and Efficient Foundation Language Models

Abstract
We introduce LLaMA, a collection of foundation
language models ranging from 7B to 65B
parameters. We train our models on trillions
of tokens, and show that it is possible to train
state-of-the-art models using publicly available
datasets exclusively, without resorting
to proprietary and inaccessible datasets. In
particular, LLaMA-13B outperforms GPT-3
(175B) on most benchmarks, and LLaMA-
65B is competitive with the best models,
Chinchilla-70B and PaLM-540B. We release
all our models to the research community1.
1 Introduction
Large Languages Models (LLMs) trained on massive
corpora of texts have shown their ability to perform
new tasks from textual instructions or from a
few examples (Brown et al., 2020). These few-shot
properties first appeared when scaling models to a
sufficient size (Kaplan et al., 2020), resulting in a
line of work that focuses on further scaling these
models (Chowdhery et al., 2022; Rae et al., 2021).
These efforts are based on the assumption that
more parameters will lead to better performance.
However, recent work from Hoffmann et al. (2022)
shows that, for a given compute budget, the best
performances are not achieved by the largest models,
but by smaller models trained on more data.
The objective of the scaling laws from Hoffmann
et al. (2022) is to determine how to best
scale the dataset and model sizes for a particular
training compute budget. However, this objective
disregards the inference budget, which becomes
critical when serving a language model at scale.
In this context, given a target level of performance,
the preferred model is not the fastest to train but the
fastest at inference, and although it may be cheaper
to train a large model to reach a certain level of
performance, a smaller one trained longer will
ultimately be cheaper at inference. For instance,
although Hoffmann et al. (2022) recommends
training a 10B model on 200B tokens, we find
that the performance of a 7B model continues to
improve even after 1T tokens.
The focus of this work is to train a series of
language models that achieve the best possible performance
at various inference budgets, by training
on more tokens than what is typically used. The
resulting models, called LLaMA, ranges from 7B
to 65B parameters with competitive performance
compared to the best existing LLMs. For instance,
LLaMA-13B outperforms GPT-3 on most benchmarks,
despite being 10  smaller. We believe that
this model will help democratize the access and
study of LLMs, since it can be run on a single GPU.
At the higher-end of the scale, our 65B-parameter
model is also competitive with the best large language
models such as Chinchilla or PaLM-540B.
Unlike Chinchilla, PaLM, or GPT-3, we only
use publicly available data, making our work compatible
with open-sourcing, while most existing
models rely on data which is either not publicly
available or undocumented (e.g. “Books – 2TB” or
“Social media conversations”). There exist some
exceptions, notably OPT (Zhang et al., 2022),
GPT-NeoX (Black et al., 2022), BLOOM (Scao
et al., 2022) and GLM (Zeng et al., 2022), but none
that are competitive with PaLM-62B or Chinchilla.
In the rest of this paper, we present an overview
of the modifications we made to the transformer
architecture (Vaswani et al., 2017), as well as our
training method. We then report the performance of
our models and compare with others LLMs on a set
of standard benchmarks. Finally, we expose some
of the biases and toxicity encoded in our models,
using some of the most recent benchmarks from
the responsible AI community.

2 Approach
Our training approach is similar to the methods
described in previous work (Brown et al., 2020;
Chowdhery et al., 2022), and is inspired by the
Chinchilla scaling laws (Hoffmann et al., 2022).
We train large transformers on a large quantity of
textual data using a standard optimizer.
2.1 Pre-training Data
Our training dataset is a mixture of several sources,
reported in Table 1, that cover a diverse set of domains.
For the most part, we reuse data sources
that have been leveraged to train other LLMs, with
the restriction of only using data that is publicly
available, and compatible with open sourcing. This
leads to the following mixture of data and the percentage
they represent in the training set:
English CommonCrawl [67%]. We preprocess
five CommonCrawl dumps, ranging from 2017
to 2020, with the CCNet pipeline (Wenzek et al.,
2020). This process deduplicates the data at the
line level, performs language identification with
a fastText linear classifier to remove non-English
pages and filters low quality content with an ngram
language model. In addition, we trained a
linear model to classify pages used as references
in Wikipedia v.s. randomly sampled pages, and
discarded pages not classified as references.
C4 [15%]. During exploratory experiments, we
observed that using diverse pre-processed CommonCrawl
datasets improves performance. We thus
included the publicly available C4 dataset (Raffel
et al., 2020) in our data. The preprocessing of C4
also contains deduplication and language identification
steps: the main difference with CCNet is
the quality filtering, which mostly relies on heuristics
such as presence of punctuation marks or the
number of words and sentences in a webpage.
Github [4.5%]. We use the public GitHub
dataset available on Google BigQuery. We only
kept projects that are distributed under the Apache,
BSD and MIT licenses. Additionally, we filtered
low quality files with heuristics based on the line
length or proportion of alphanumeric characters,
and removed boilerplate, such as headers, with regular
expressions. Finally, we deduplicate the resulting
dataset at the file level, with exact matches.
Wikipedia [4.5%]. We add Wikipedia dumps
from the June-August 2022 period, covering 20
languages, which use either the Latin or Cyrillic
scripts: bg, ca, cs, da, de, en, es, fr, hr, hu, it,
nl, pl, pt, ro, ru, sl, sr, sv, uk. We process the
data to remove hyperlinks, comments and other
formatting boilerplate.
Gutenberg and Books3 [4.5%]. We include
two book corpora in our training dataset: the Gutenberg
Project, which contains books that are in the
public domain, and the Books3 section of ThePile
(Gao et al., 2020), a publicly available dataset
for training large language models. We perform
deduplication at the book level, removing books
with more than 90% content overlap.
ArXiv [2.5%]. We process arXiv Latex files
to add scientific data to our dataset. Following
Lewkowycz et al. (2022), we removed everything
before the first section, as well as the bibliography.
We also removed the comments from the .tex files,
and inline-expanded definitions and macros written
by users to increase consistency across papers.
Stack Exchange [2%]. We include a dump of
Stack Exchange, a website of high quality questions
and answers that covers a diverse set of domains,
ranging from computer science to chemistry.
We kept the data from the 28 largest websites, removed
the HTML tags from text and sorted the
answers by score (from highest to lowest).
Tokenizer. We tokenize the data with the bytepair
encoding (BPE) algorithm (Sennrich et al.,
2015), using the implementation from Sentence-
Piece (Kudo and Richardson, 2018). Notably, we
split all numbers into individual digits, and fallback
to bytes to decompose unknown UTF-8 characters.
Overall, our entire training dataset contains
roughly 1.4T tokens after tokenization. For most of
our training data, each token is used only once during
training, with the exception of the Wikipedia
and Books domains, over which we perform approximately
two epochs.
2.2 Architecture
Following recent work on large language models,
our network is based on the transformer architecture
(Vaswani et al., 2017). We leverage various
improvements that were subsequently proposed,
and used in different models such as PaLM. Here
are the main difference with the original architecture,
and where we were found the inspiration for
this change (in bracket):
Pre-normalization [GPT3]. To improve the
training stability, we normalize the input of each
transformer sub-layer, instead of normalizing the
output. We use the RMSNorm normalizing function,
introduced by Zhang and Sennrich (2019).
SwiGLU activation function [PaLM]. We replace
the ReLU non-linearity by the SwiGLU activation
function, introduced by Shazeer (2020) to
improve the performance. We use a dimension of
2
34d instead of 4d as in PaLM.
Rotary Embeddings [GPTNeo]. We remove the
absolute positional embeddings, and instead, add
rotary positional embeddings (RoPE), introduced
by Su et al. (2021), at each layer of the network.
The details of the hyper-parameters for our different
models are given in Table 2.

2.3 Optimizer
Our models are trained using the AdamW optimizer
(Loshchilov and Hutter, 2017), with the following
hyper-parameters:  1 = 0:9;  2 = 0:95.
We use a cosine learning rate schedule, such that
the final learning rate is equal to 10% of the maximal
learning rate. We use a weight decay of 0:1 and
gradient clipping of 1:0. We use 2; 000 warmup
steps, and vary the learning rate and batch size with
the size of the model (see Table 2 for details).
2.4 Efficient implementation
We make several optimizations to improve the training
speed of our models. First, we use an efficient
implementation of the causal multi-head attention
to reduce memory usage and runtime. This implementation,
available in the xformers library,2 is
inspired by Rabe and Staats (2021) and uses the
backward from Dao et al. (2022). This is achieved
by not storing the attention weights and not computing
the key/query scores that are masked due to
the causal nature of the language modeling task.
To further improve training efficiency, we reduced
the amount of activations that are recomputed
during the backward pass with checkpointing.
More precisely, we save the activations that
are expensive to compute, such as the outputs of
linear layers. This is achieved by manually implementing
the backward function for the transformer
layers, instead of relying on the PyTorch autograd.
To fully benefit from this optimization, we need to
reduce the memory usage of the model by using
model and sequence parallelism, as described by
Korthikanti et al. (2022). Moreover, we also overlap
the computation of activations and the communication
between GPUs over the network (due to
all_reduce operations) as much as possible.
When training a 65B-parameter model, our code
processes around 380 tokens/sec/GPU on 2048
A100 GPU with 80GB of RAM. This means that
training over our dataset containing 1.4T tokens
takes approximately 21 days.

3 Main results
Following previous work (Brown et al., 2020), we
consider zero-shot and few-shot tasks, and report
results on a total of 20 benchmarks:
• Zero-shot. We provide a textual description
of the task and a test example. The model
either provides an answer using open-ended
generation, or ranks the proposed answers.
• Few-shot. We provide a few examples of the
task (between 1 and 64) and a test example.
The model takes this text as input and generates
the answer or ranks different options.
We compare LLaMA with other foundation models,
namely the non-publicly available language
models GPT-3 (Brown et al., 2020), Gopher (Rae
et al., 2021), Chinchilla (Hoffmann et al., 2022)
and PaLM (Chowdhery et al., 2022), as well as
the open-sourced OPT models (Zhang et al., 2022),
GPT-J (Wang and Komatsuzaki, 2021), and GPTNeo
(Black et al., 2022). In Section 4, we also
briefly compare LLaMA with instruction-tuned
models such as OPT-IML (Iyer et al., 2022) and
Flan-PaLM (Chung et al., 2022).

We evaluate LLaMA on free-form generation
tasks and multiple choice tasks. In the multiple
choice tasks, the objective is to select the most
appropriate completion among a set of given options,
based on a provided context. We select the
completion with the highest likelihood given the
provided context. We follow Gao et al. (2021)
and use the likelihood normalized by the number
of characters in the completion, except for certain
datasets (OpenBookQA, BoolQ), for which we follow
Brown et al. (2020), and select a completion
based on the likelihood normalized by the likelihood
of the completion given “Answer:” as context:

