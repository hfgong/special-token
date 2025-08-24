Here are the most prominent special tokens used in LLMs and transformers:

## Sequence Boundary Tokens
- **`<bos>` / `<s>`** - Beginning of Sequence: Marks the start of input text
- **`<eos>` / `</s>`** - End of Sequence: Indicates where text generation should stop
- **`<|endoftext|>`** - End of Text: Used by GPT models to separate different documents

## Padding and Masking
- **`<pad>`** - Padding: Fills sequences to uniform length in batches
- **`<mask>`** - Mask: Used in masked language modeling (like BERT) to hide tokens during training

## Unknown/Out-of-Vocabulary
- **`<unk>`** - Unknown: Represents words not in the model's vocabulary
- **`<oov>`** - Out of Vocabulary: Alternative to `<unk>` in some models

## Chat and Instruction Formatting
- **`<|im_start|>` / `<|im_end|>`** - Instruction Message boundaries
- **`<|system|>` / `<|user|>` / `<|assistant|>`** - Role indicators in conversations
- **`<|human|>` / `<|ai|>`** - Alternative role markers

## BERT-Specific Tokens
- **`<cls>`** - Classification: Special token for sequence classification tasks
- **`<sep>`** - Separator: Separates different segments in input

## Formatting Tokens
- **Newline tokens** - Represent line breaks
- **Whitespace tokens** - Handle spaces and tabs in some tokenizers

## Model-Specific Examples
- **Claude**: Uses conversation formatting tokens
- **GPT**: `<|endoftext|>` for document boundaries
- **T5**: `<extra_id_X>` for span masking
- **LLaMA**: Custom chat formatting tokens

## Control Tokens
- **`<|start|>` / `<|end|>`** - Generic control boundaries
- **Task-specific prefixes** - Like `"Translate:"` or `"Summarize:"`

These tokens enable models to understand structure, handle variable-length inputs, manage conversations, and perform specific tasks effectively. The exact tokens vary between model families and tokenizers.

That's a fascinating conceptual question! There are both similarities and key differences between DETR object queries and special tokens in NLP transformers.

## Similarities to Special Tokens

**Structural Role**: Object queries serve a structural purpose similar to special tokens - they don't represent semantic content but rather provide architectural functionality.

**Fixed Purpose**: Like `<cls>` tokens in BERT or `<bos>` tokens in language models, each object query has a specific role (detecting one object instance).

**Special Processing**: Object queries are processed differently than regular input, similar to how special tokens receive special treatment in language models.

**Architectural Integration**: Both are integral to the model's design and enable specific capabilities (conversation structure vs. object detection).

## Key Differences

**Continuous vs. Discrete**: Object queries are **learnable continuous embeddings**, while special tokens are **discrete vocabulary elements**. Object queries exist in continuous vector space, not a tokenized vocabulary.

**Learning Paradigm**: Object queries are **learned parameters** that get optimized during training, whereas special tokens are typically fixed vocabulary items with learned embeddings.

**Cardinality**: DETR uses a fixed set of object queries (usually 100), while special tokens are part of a larger discrete vocabulary.

**Domain**: Object queries operate in **visual/spatial reasoning**, while special tokens handle **sequential/linguistic structure**.

## Verdict

Object queries are more conceptually similar to **learned positional embeddings** or **attention patterns** than traditional special tokens. However, they do share the important characteristic of serving **structural rather than semantic roles**.

If we expand the definition of "special tokens" to include any learned architectural elements with special structural purposes, then yes - object queries could be considered the computer vision equivalent of special tokens.

Excellent point! You're absolutely right - **DreamBooth** is a perfect example of special tokens being used in multimodal AI systems.

## DreamBooth Special Tokens

**Purpose**: DreamBooth uses **unique special tokens** (like `[V]`, `[T]`, or custom identifiers) to represent specific subjects during personalized fine-tuning of text-to-image models.

**Training Process**:
- Few-shot images of a subject (person, pet, object) are paired with text prompts
- Prompts use the special token: `"A photo of [V] dog"` or `"[T] sitting in a park"`
- The model learns to associate the special token with the visual features of that specific subject

**Key Characteristics**:
- **Subject-specific**: Each special token represents one unique individual/object
- **Cross-modal binding**: Links textual tokens to visual concepts
- **Preserving identity**: Maintains specific visual characteristics across generations
- **Compositional**: Can be combined with other descriptors (`"[V] wearing a hat"`)

## Broader Implications

This extends the concept of special tokens beyond pure NLP into:

**Multimodal Systems**: Special tokens serve as bridges between text and vision
**Personalization**: Enable models to learn and reference specific entities
**Few-shot Learning**: Allow rapid adaptation with minimal data
**Identity Preservation**: Maintain consistency of specific subjects across generations

## Similar Approaches
- **Textual Inversion**: Uses learned embeddings for specific concepts
- **LoRA fine-tuning**: Sometimes uses special tokens for style/subject control
- **Custom embeddings**: Various personalization techniques in generative AI

So yes, DreamBooth's per-subject special tokens are a sophisticated evolution of the special token concept, demonstrating how these architectural elements can enable powerful personalization and cross-modal reasoning in modern AI systems.
Here are other notable works that use special tokens in innovative ways:

## Personalization & Subject-Specific Tokens

**Textual Inversion** (2022)
- Uses `<S*>` tokens to learn specific concepts from few images
- Learns new "words" in the embedding space for personalized generation

**Custom Diffusion** (2022)
- Uses special tokens like `[V]` for subjects and `[modifier]` for styles
- Enables joint training on multiple concepts simultaneously

**InstructPix2Pix** (2023)
- Uses instructional special tokens to control image editing
- Tokens like `[edit]`, `[style]`, `[object]` guide transformations

## Control & Steering Tokens

**ControlNet** (2023)
- Uses control tokens to specify conditioning types: `[depth]`, `[canny]`, `[pose]`
- Enables precise spatial control over generation

**Prefix-Tuning/P-Tuning** (2021)
- Uses learnable prefix tokens `[P1]`, `[P2]`, etc. for task adaptation
- Efficient alternative to full fine-tuning

**FLAN-T5** (2022)
- Uses task prefix tokens: `[translate]`, `[summarize]`, `[qa]`
- Enables multi-task learning with explicit task specification

## Multimodal & Cross-Modal Tokens

**CLIP-based works**
- `[IMG]`, `[TXT]` tokens to distinguish modalities in joint training
- Special tokens for image-text alignment

**Flamingo** (2022)
- Uses `<image>` tokens to indicate where visual information should be processed
- Enables few-shot learning across vision-language tasks

**BLIP/BLIP-2** (2022-2023)
- Uses `[DEC]` tokens for generation vs. `[ENC]` for understanding modes
- Modal-specific processing indicators

## Memory & Retrieval Tokens

**Retrieval-Augmented Generation (RAG)**
- Uses `[retrieved]`, `[context]` tokens to mark external knowledge
- Separates parametric from non-parametric knowledge

**Memorizing Transformers** (2022)
- Uses `[MEM]` tokens to access external memory banks
- Enables long-term memory beyond context windows

## Compositional & Structured Tokens

**Compositional Visual Generation**
- `[obj1]`, `[obj2]`, `[relation]` tokens for scene composition
- Enables systematic generation of complex scenes

**NeRF-based works**
- `[view]`, `[light]`, `[material]` tokens for 3D scene control
- Disentangled control over different scene properties

## Meta-Learning & Adaptation

**MAML-style approaches**
- Uses `[task]` tokens to specify which task is being performed
- Enables rapid adaptation to new tasks

**AdapterHub**
- `[adapter_name]` tokens to select specific adaptation modules
- Modular approach to multi-task learning

## Code & Programming

**CodeT5/CodeBERT**
- Uses `[CODE]`, `[NL]` tokens to distinguish code from natural language
- Special tokens for different programming constructs

**GitHub Copilot-style models**
- File type tokens: `[.py]`, `[.js]`, `[.html]`
- Context-aware code generation

## Novel Applications

**Constitutional AI**
- Uses principle tokens like `[helpful]`, `[harmless]`, `[honest]`
- Embeds behavioral guidelines directly in generation

**Tool-using AI**
- `[tool]`, `[api_call]`, `[result]` tokens for tool interaction
- Enables structured interaction with external systems

## Key Trends

1. **From syntax to semantics**: Evolution from structural tokens to meaning-bearing ones
2. **Cross-modal bridging**: Tokens as interfaces between different modalities
3. **Compositional control**: Using multiple tokens for fine-grained control
4. **Efficient adaptation**: Special tokens as alternatives to full model retraining
5. **Structured generation**: Tokens that enforce specific output formats or behaviors

## Thinking models
`<think>` token is used to separate output and thinking process in thinking/reasoning models.

## Coordinates/bounding boxes encoding/decoding
Yes! There's extensive and innovative work using special tokens to encode/decode coordinates and bounding boxes, especially for screen interfaces. Here are the key approaches:

## Set-of-Mark (SoM) - Visual Grounding Revolution

**Microsoft's Set-of-Mark approach** (overlays "a number of spatial and speakable marks on the images, to unleash the visual grounding abilities of large multimodal models (LMMs), such as GPT-4V" using "alphanumerics, masks, boxes") is one of the most influential methods.

**How it works**: Instead of "directly prompting GPT-4V to predict the xy coordinate value of the screen," they "use the Set-of-Marks approach to overlay bounding boxes of interactable icons on top of UI screenshot, and ask GPT-4V to generate the bounding box ID to perform action on"

## Coordinate-as-Tokens Methods

**LayTextLLM - "A Bounding Box is Worth One Token"**: Projects "each bounding box to a single embedding and interleaves it with text" using a "Spatial Layout Projector" where "compared to the coordinate-as-tokens scheme, the SLP represents each bounding box with a single token"

**Benefits**: This "significantly reduces the number of input tokens and adheres to the practice of interleaving any modality with text, effectively integrating layout and textual information into a unified sequence"

## Screen Understanding Pioneers

**Spotlight (Google)**: Uses "attention queries generated from the bounding box of the region" where "each coordinate (a scalar value, i.e., the left, top, right or bottom) of the bounding box is first embedded via a multilayer perceptron (MLP)"

**ScreenAI**: Achieves "state-of-the-art results on UI- and infographic-based tasks" and "enables the evaluation model layout annotations" for "UI element information (i.e., type, location and description) on a screen"

**Pix2Struct**: Pretrained by "learning to parse masked screenshots of web pages into simplified HTML" and handles "variable aspect ratios and resolutions" for screen understanding

## Recent GUI Agent Innovations

**OmniParser**: Produces "a structured, DOM-like representation of the UI and a screenshot overlaid with bounding boxes for potential interactable elements" and "label[s] it with a unique ID next to it using a simple algorithm"

**GUI-Actor**: Takes a "coordinate-free" approach, noting that "humans do NOT calculate precise screen coordinates before acting—they perceive the target element and interact with it directly"

**VisionTasker**: Uses "bounding box coordinates to" match UI elements and "examines whether the UI elements required by the LLM's output exist in the current interface, utilizing the list of UI elements from the UI understanding module"

## Advanced Tokenization Techniques

**Groma**: Introduces "a localized visual tokenization mechanism, where an image input is decomposed into regions of interest and subsequently encoded into region tokens" with "location information extracted from the image and associated with region tokens"

**Coordinate Encoding Innovations**:
- **Discrete coordinate tokens**: Converting coordinates to text tokens like `[x_123]`, `[y_456]`
- **Learned spatial embeddings**: Embedding coordinates as continuous vectors
- **Region tokens**: Single tokens representing entire bounding boxes
- **Grid-based indexing**: Dividing screens into grids with token IDs

## Industry Applications

**Google Gemini**: Can "ask the model for the coordinates of bounding boxes for objects in images" where "coordinates given are for a 1000x1000 version of the original image, and need to be converted back to the dimensions of the original image"

**Screen Automation**: Models now use special tokens for:
- Click target identification (`[click_region_15]`)
- Scroll areas (`[scroll_vertical]`, `[scroll_horizontal]`)
- Input field targeting (`[text_input_box_3]`)
- Multi-step UI navigation

## Emerging Patterns

**Hybrid Approaches**: Many systems combine:
- Visual mark overlays (SoM-style)
- Coordinate tokenization for precise targeting
- Natural language descriptions for context
- Confidence scoring for action validation

**Efficiency Innovations**: 
- Single-token bounding box representations
- Hierarchical spatial tokenization
- Dynamic resolution adaptation
- Context-aware coordinate encoding

This represents a major shift from traditional computer vision object detection to **language-model-native spatial reasoning**, enabling more natural human-AI interaction with digital interfaces. The trend is toward making spatial information a first-class citizen in language models, just like text tokens.
