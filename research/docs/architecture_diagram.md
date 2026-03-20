# Architecture Diagram — Frequency-Aware Dual-Branch Fusion

## Overview Architecture

```mermaid
graph TD
    A["Source A<br/>(CT/PET/SPECT)"] --> NSST_A["NSST<br/>Decomposition"]
    B["Source B<br/>(MRI)"] --> NSST_B["NSST<br/>Decomposition"]

    NSST_A --> HF_A["HF_A<br/>(edges, texture)"]
    NSST_A --> LF_A["LF_A<br/>(structure, intensity)"]
    NSST_B --> HF_B["HF_B<br/>(edges, texture)"]
    NSST_B --> LF_B["LF_B<br/>(structure, intensity)"]

    HF_A --> HF_BRANCH["🔺 HF Branch<br/>CNN + Mamba Hybrid"]
    HF_B --> HF_BRANCH

    LF_A --> LF_BRANCH["🔵 LF Branch<br/>Mamba + Modality Balance"]
    LF_B --> LF_BRANCH

    HF_BRANCH --> HF_F["HF_fused"]
    LF_BRANCH --> LF_F["LF_fused"]

    HF_F --> MERGE["Inverse NSST<br/>+ Refinement CNN"]
    LF_F --> MERGE

    MERGE --> F["Fused Image F"]

    style A fill:#ff9999,stroke:#cc0000
    style B fill:#99ccff,stroke:#0066cc
    style F fill:#99ff99,stroke:#00cc00,stroke-width:3px
    style HF_BRANCH fill:#ffe0b2,stroke:#e65100
    style LF_BRANCH fill:#b3e5fc,stroke:#0277bd
    style NSST_A fill:#f5f5f5,stroke:#999
    style NSST_B fill:#f5f5f5,stroke:#999
    style MERGE fill:#e8f5e9,stroke:#2e7d32
```

## HF Branch Detail — CNN + Mamba Hybrid

```mermaid
graph LR
    HF_A["HF_A"] --> CONCAT["Concat"]
    HF_B["HF_B"] --> CONCAT

    CONCAT --> CNN["Local CNN<br/>3×3 Conv × 3<br/>(edge refinement)"]
    CNN --> SOBEL["Sobel-Guided<br/>Attention Map"]
    CNN --> MAMBA_HF["Cross-Modal<br/>Mamba Block<br/>(long-range consistency)"]
    SOBEL --> MUL["⊗"]
    MAMBA_HF --> MUL

    MUL --> HF_F["HF_fused"]

    style CNN fill:#ffe0b2,stroke:#e65100
    style MAMBA_HF fill:#fff3e0,stroke:#ff6f00
    style SOBEL fill:#fce4ec,stroke:#c62828
    style HF_F fill:#fff9c4,stroke:#f9a825
```

## LF Branch Detail — Mamba + Modality Balance

```mermaid
graph LR
    LF_A["LF_A"] --> ENC_A["Mamba<br/>Encoder"]
    LF_B["LF_B"] --> ENC_B["Mamba<br/>Encoder"]

    ENC_A --> BAL["Modality<br/>Balance Module<br/>(saliency-weighted)"]
    ENC_B --> BAL

    BAL --> DEC["Mamba<br/>Decoder"]
    DEC --> LF_F["LF_fused"]

    style ENC_A fill:#b3e5fc,stroke:#0277bd
    style ENC_B fill:#b3e5fc,stroke:#0277bd
    style BAL fill:#e1bee7,stroke:#6a1b9a
    style DEC fill:#b3e5fc,stroke:#0277bd
    style LF_F fill:#fff9c4,stroke:#f9a825
```

## Modality Balance Module Detail

```mermaid
graph TD
    FA["Feature_A"] --> VAR_A["σ²_A<br/>(local variance)"]
    FB["Feature_B"] --> VAR_B["σ²_B<br/>(local variance)"]

    VAR_A --> LAMBDA["λ = σ²_A / (σ²_A + σ²_B)<br/>(saliency weight)"]
    VAR_B --> LAMBDA

    FA --> WEIGHTED["F_balanced = λ × F_A + (1-λ) × F_B"]
    FB --> WEIGHTED
    LAMBDA --> WEIGHTED

    WEIGHTED --> OUT["Balanced LF Features"]

    style LAMBDA fill:#e1bee7,stroke:#6a1b9a
    style WEIGHTED fill:#f3e5f5,stroke:#7b1fa2
```

## Loss Function

```mermaid
graph LR
    F["F (fused)"] --> L1["L_intensity<br/>‖F − max(A,B)‖₁"]
    F --> L2["L_texture<br/>‖∇F − max(∇A,∇B)‖₁"]
    F --> L3["L_ssim<br/>1 − ½(SSIM_AF + SSIM_BF)"]
    F --> L4["L_perceptual<br/>CSF-weighted distortion"]

    L1 --> TOTAL["L_total = α₁L₁ + α₂L₂ + α₃L₃ + α₄L₄"]
    L2 --> TOTAL
    L3 --> TOTAL
    L4 --> TOTAL

    L1 -.- T1["→ VIF↑ SCD↑"]
    L2 -.- T2["→ Q^AB/F↑ Q_SF↑"]
    L3 -.- T3["→ MS-SSIM↑ Q_Y↑"]
    L4 -.- T4["→ Q_CV↑ Q_CB↑"]

    style TOTAL fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style L4 fill:#fff3e0,stroke:#e65100
```

## So sánh với CDDFuse (baseline)

```mermaid
graph TD
    subgraph CDDFuse["CDDFuse (baseline)"]
        direction TB
        CD_IN["A, B"] --> CD_INN["INN<br/>Decomposition"]
        CD_INN --> CD_HF["Detail Branch<br/>Cross-Attention<br/>Transformer"]
        CD_INN --> CD_LF["Base Branch<br/>Simple CNN"]
        CD_HF --> CD_MERGE["INN Inverse"]
        CD_LF --> CD_MERGE
        CD_MERGE --> CD_F["F"]
    end

    subgraph OURS["Proposed (ours)"]
        direction TB
        OUR_IN["A, B"] --> OUR_NSST["NSST<br/>Decomposition"]
        OUR_NSST --> OUR_HF["HF Branch<br/>CNN + Mamba<br/>+ Sobel Attention"]
        OUR_NSST --> OUR_LF["LF Branch<br/>Mamba Encoder-Decoder<br/>+ Modality Balance"]
        OUR_HF --> OUR_MERGE["Inverse NSST<br/>+ Refinement"]
        OUR_LF --> OUR_MERGE
        OUR_MERGE --> OUR_F["F"]
    end

    style CDDFuse fill:#fff3e0,stroke:#e65100
    style OURS fill:#e8f5e9,stroke:#2e7d32

    style CD_LF fill:#ffcdd2,stroke:#c62828
    style OUR_LF fill:#c8e6c9,stroke:#2e7d32
    style CD_HF fill:#ffcdd2,stroke:#c62828
    style OUR_HF fill:#c8e6c9,stroke:#2e7d32
```
