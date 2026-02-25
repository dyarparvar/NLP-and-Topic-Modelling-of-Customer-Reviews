# NLP-and-Topic-Modelling-of-Customer-Reviews
**Darya Yarparvar** | November 2025
______________________________________________________________

## Problem Statement
**********’s customer-centric value proposition, centred on affordable, flexible, 24/7 fitness access, has established its market leadership. This project aims to provide a systematic approach for extracting concerns expressed in customer reviews using advanced NLP techniques. By transforming unstructured text into clear, actionable insights, the analysis can inform service delivery and member experience improvements and support ********** in maintaining its market-leading position.

## Dataset
The data were provided from two different platforms (Table 1).

<img width="865" height="235" alt="image" src="https://github.com/user-attachments/assets/fc1eb95f-706a-44cb-bfce-5bf7de38ee9f" />

Table 1. Overview of the datasets (data provided by **********).

## Data Cleaning and Pre-Processing
Duplicated records were removed along with irrelevant features (review_title, review_source, domain_url, webshop_name, unit_id, tags, location_id, user_id, reply_date, review_id, review_date, user_name, social_source) and entries with missing reviews. Three features were retained: 
- location name
- review content
- review stars (<3 stars classified as negative)
Semantic duplicates in location names were minimised, with missing names retained as NaN.
For initial exploratory analysis, reviews were cleaned by removing HTML tags, digits, punctuation, and non-ASCII characters, then tokenised using NLTK. Stop words were combined across languages, accepted as sufficient for preliminary analysis. For the main topic modeling, these preprocessing steps were unnecessary as modern language models (Phi4) handle this directly. A fixed random seed and other necessary measures were taken to maximise reproducibility.

## Exploratory Data Analysis
We began by analysing all reviews across both platforms to establish baseline patterns. Filtering for negative reviews (21% of total reviews) revealed distinct pain points as positive sentiment words disappeared from word frequency distributions, replaced by complaint-related terms such as "equipment/machines", "membership", "time", and "people/staff/email" (Fig. 1, 2).

<img width="823" height="539" alt="image" src="https://github.com/user-attachments/assets/6dfca471-e488-47e5-96d4-c6c980d84452" />

Figure 1. Word frequency distribution displaying the most frequent terms in all reviews. Positive sentiment words ("good", "great", "friendly", "clean") dominate both platforms, with "equipment" appearing as the most frequent term. Venn diagram shows 9 words appearing in both platforms' high-frequency lists validating the merged dataset approach for capturing recurring themes. Some low-value terms such as "really" remain in the frequency list. Further refinement of the custom stop-word list would be beneficial to increase the signal-to-noise ratio. While the bar chart provides a precise quantitative ranking, the word cloud’s internal preprocessing and layout algorithm offer a qualitative overview where spatial fit and word length can skew the perceived dominance of terms.

<img width="878" height="596" alt="image" src="https://github.com/user-attachments/assets/64eba3a6-386f-4470-9367-cc6a39a3c9ff" />

Figure 2. Word frequency distribution displaying the most frequent terms in negative reviews (reviews with review_stars < 3). After filtering for negative sentiment, positive words disappear and the remaining terms now reflect complaints. "Equipment" dominates both platforms, followed by "membership" (Trustpilot) and "staff/people" (Google), with similar frequency patterns. Venn diagram shows 8 words appearing in both platforms' high-frequency lists, with 2 words unique to each platform. This substantial overlap in complaint vocabulary indicates consistent pain points across review sources and validates the merged dataset approach for capturing recurring customer dissatisfaction themes. Some low-value terms such as "get" remain in the frequency list. Further refinement of the custom stop-word list would be beneficial to increase the signal-to-noise ratio. While the bar chart provides a precise quantitative ranking, the word cloud’s internal preprocessing and layout algorithm offer a qualitative overview where spatial fit and word length can skew the perceived dominance of terms.

Initially, we focused on locations common to both datasets. However, only 56% of locations appeared in both platforms (corresponding to ~64% of negative reviews), risking exclusion of ~36% of complaints (Fig. 3). When examining worst-performing locations (top 20 by negative review volume) within each platform separately, only 7 overlapped. Therefore, we merged both datasets and identified 30 worst-performing locations based on combined negative review counts (Table 2), representing ~36% of all negative reviews. Word frequency analysis revealed "membership", "equipment", "time", "people”, “staff", and "email" as primary pain points (Fig. 4).

<img width="643" height="318" alt="image" src="https://github.com/user-attachments/assets/a8e01509-299a-4ada-af44-7cada0b2f70a" />

Figure 3. Distribution of gym locations across review platforms. Left: Venn diagram showing overlap of locations across Trustpilot and Google platforms; Only 56% of locations appear across both datasets. Right: Venn diagram showing overlap of worst-performing locations across Trustpilot and Google platforms; Only 7 locations appear across both datasets.

<img width="860" height="695" alt="image" src="https://github.com/user-attachments/assets/ec19921f-dba1-4a6a-b676-ec1aaad2d624" />
Table 2. Overall 30 worst-performing locations by volume of negative reviews. The table ranks sites with the highest counts of negative review, highlighting a strong concentration in London (17 locations) and smaller number of locations in Birmingham and other cities. The entry labelled as “345” reflects a malformed location name. 1291 “Unknown” labeled negative reviews lack location information, limiting the ability to link complaints to specific locations.

<img width="856" height="306" alt="image" src="https://github.com/user-attachments/assets/2d0bff4d-c443-49fa-8528-04c6f447e09a" />

Figure 4. Word frequency distribution displaying the most frequent terms in negative reviews across the 30 worst-performing gym locations. "Membership" emerges as the dominant term, followed by issues related to "equipment", “time”, “people/staff”, and “email” indicating primary pain points at problematic sites. Some low-value terms such as "get" remain in the frequency list. Further refinement of the custom stop-word list would be beneficial to increase the signal-to-noise ratio. While the bar chart provides a precise quantitative ranking, the word cloud’s internal preprocessing and layout algorithm offer a qualitative overview where spatial fit and word length can skew the perceived dominance of terms.

To systematically extract recurring themes, we applied BERTopic modeling, mapping outliers (~42% for common locations and ~33% for worst locations) to similar clusters.  BERTopic, as an embedding-based neural topic modelling framework, is particularly well suited for extracting semantically coherent and meaningful themes from short, noisy customer review data (Krishnan, 2023). Considering the low non-English review proportion (<0.01%), we used standard BERTopic. For common locations, we identified air conditioning/temperature, equipment availability, and membership management as top systemic issues (Table 3). For worst-performing locations, cleanliness, staff behaviour, and membership/app issues dominated, followed by machines/weights and customer service (Table 4). These findings are in agreement with word frequency patterns (Fig. 4), validating consistency between lexical and topic-based approaches.


<img width="769" height="738" alt="image" src="https://github.com/user-attachments/assets/2c22232c-34a6-4499-8472-4d6ad39cfb14" />

Table 3. Dominant complaint themes in negative reviews from locations common to both datasets. Air conditioning and heat problems, equipment shortages, and membership or app-related issues emerge as the leading pain points, followed by overcrowding, poor facility maintenance, class scheduling problems, staff conduct, access, and hygiene issues. Percentages indicate each theme’s contribution to the overall volume of negative reviews in common locations.

For worst-performing locations specifically, cleanliness of facilities (6.7%), staff behavior (6.0%), and membership/app issues (5.1%) dominated complaints, followed closely by overcrowding (5.0%) and customer service responsiveness (4.7%) (Table 4). These findings align with word frequency patterns observed during exploratory analysis (Fig. 4), validating consistency between word-based and topic-based approaches.

<img width="766" height="678" alt="image" src="https://github.com/user-attachments/assets/02f87f52-aaa7-4dcb-a6aa-c6cabe0b0d97" />


Table 4. Dominant complaint themes in negative reviews from the worst-performing locations. Hygiene issues in toilets and changing areas, unprofessional staff behaviour, and membership or app-related problems are the leading drivers of poor experience. Percentages denote each theme’s contribution to the total volume of negative reviews in these locations.


## Modelling
### Emotion Analysis
We applied a pre-trained BERT-based emotion classification to categorise reviews into six emotions, focusing on anger-expressing reviews to identify acute pain points (Table 5). We then applied BERTopic modeling, mapping outliers (~40%) to similar clusters.

### Topic Modelling
As a generative topic modelling approach, we used Phi4, a 14‑billion‑parameter high-performance SLM (Garg, 2025), leveraging its reasoning capabilities for semantic topic extraction rather than traditional probabilistic methods (Mu et al., 2024). It extracted the top three topics per negative review. A multilingual BERTopic consolidated these and reduced them to 200 coherent themes without mapping outliers (17%) in order to preserve topic purity. BERTopic’s native outlier detection is retained to provide a different view of the data than LDA, where all documents must be assigned.
As a methodological baseline, we applied Latent Dirichlet Allocation (LDA) to extract 10 topics, assessing whether neural models captured substantially different patterns than classical probabilistic methods (Table 5). 
### Actionable Insights
Phi4 transformed identified topics into actionable insights for organisational teams, shifting analysis from describing complaints to defining interventions (Table 6).

<img width="886" height="715" alt="image" src="https://github.com/user-attachments/assets/c306877e-903d-434f-b8d7-d975b6b762a3" />
<img width="887" height="431" alt="image" src="https://github.com/user-attachments/assets/0359380d-86ff-4702-80cb-fd15f71fb16c" />


Table 5. Model configurations, parameters, and implementation settings for topic modelling.

<img width="877" height="214" alt="image" src="https://github.com/user-attachments/assets/3360d153-b6e2-4549-be59-2737da696d41" />


Table 6. Model configurations, parameters, and implementation settings for extracting actionable insights.


## Results
### Emotion Analysis
While joy dominated all reviews, anger was primary in negative reviews (44%), followed by sadness and joy (24% each), consistent across both platforms. Joy in negative reviews reflects emotion models detecting tone rather than sentiment, capturing sarcasm or praise within negative feedback (Fig. 5).
<img width="878" height="322" alt="image" src="https://github.com/user-attachments/assets/a1363af2-9926-4515-bcb9-7852eb0c2ca2" />
Figure 5. Distribution of dominant emotions across all reviews (top), negative reviews by platform (middle), and all negative reviews combined (bottom). Anger dominates both platforms, followed by sadness and joy. The similar emotional profiles across platforms indicate consistent emotional responses to negative experiences regardless of review source. Anger represents 44% of all negative reviews, followed by sadness and joy (each ~24%). Fear, love, and surprise collectively account for less than 10%. The presence of joy in negative reviews reflects emotion models detecting tone rather than overall sentiment, capturing sarcasm or praise of minor positive aspects within otherwise critical feedback. Platform-specific patterns are consistent with merged data, validating the combined approach for emotion analysis.

BERTopic analysis of anger-expressing reviews exposed multilingual processing limitations: the top category (7.4%) improperly bundled equipment, maintenance, and staff issues for non-English reviews (Table 7). For English reviews staff behaviour and hygiene dominated (combined 8.8%), followed by customer service (4.2%) and membership problems (4.0%) (Table 7).
<img width="768" height="657" alt="image" src="https://github.com/user-attachments/assets/68eba57e-11f3-4695-90f0-34b7815ee7a6" />

Table 7. Dominant complaint themes in angry reviews. Equipment quality and centre conditions, unprofessional staff behaviour, and hygiene standards are the leading drivers of dissatisfaction, followed by weak customer service, membership cancellations and refunds, access and payment issues, and class or training-related problems. Percentages denote each theme’s contribution to the total volume of angry reviews.

The angry-only key complaints suggest system-wide issues rather than location-specific problems. The worst-only key complaints represent slow-burn issues that may be tolerated temporarily but eventually drive silent churn. The overlapping key complaints might be prioritised for future interventions (Table 8).
<img width="877" height="436" alt="image" src="https://github.com/user-attachments/assets/6bf8a35b-b744-49ce-9785-fe059283141d" />

Table 8. Thematic comparison of key complaints between angry reviews and worst-performing locations, highlighting system-wide reputational risks and location-specific churn drivers.

### Topic Modelling
Phi4 extracted 3.09 topics per review (vs. target of 3), suggesting minor hallucination or splitting errors but remaining acceptable. BERTopic consolidation revealed equipment availability (5.0%) and gym atmosphere (5.0%) as dominant complaints, followed by customer service (3.7%), cleanliness (2.8%), overcrowding (2.5%), staff behaviour (2.4%), membership cancellation (2.3%), air conditioning (1.9%), showers (1.8%), and poor equipment maintenance (1.5%). Top ten themes represented 30.9% of negative reviews, validating word frequency and emotion analysis findings (Table 9).

<img width="869" height="559" alt="image" src="https://github.com/user-attachments/assets/2323ff82-e8e8-45d1-9fc7-f16b7a93320e" />
<img width="869" height="417" alt="image" src="https://github.com/user-attachments/assets/8f69050d-7432-4c9a-92e4-7f0344fc7b14" />

Table 9. Dominant complaint themes in all negative reviews. Concerns about equipment availability, poor maintenance, and centre conditions, along with negative comparisons to other gyms, hygiene issues, and overcrowding, are the main drivers of dissatisfaction, followed by unprofessional staff behaviour, membership and cancellation difficulties, air conditioning and shower issues, and equipment breakdowns. Percentages denote each theme’s contribution to the total volume of negative reviews.

LDA extracted 10 topics, with "equipment", "staff", "machines", and "membership" dominating salient terms. Facility terms ("changing", "showers", "air", "water", "dirty", "toilets", "hot", "cold") showed mid-range saliency, while access terms ("day", "pass", "code", "pin") showed moderate saliency, validating the core themes across all models (Fig. 6).

<img width="852" height="463" alt="image" src="https://github.com/user-attachments/assets/e69e2f33-a03b-4097-aa0c-536ec0377582" />

Figure 6. Global overview of topics and terms across the entire corpus extracted by the LDA model. Left: Intertopic distance map, where circle size reflects marginal topic prevalence and spatial proximity indicates semantic similarity. The separation of Topics 7–9 from the dominant Topics 1–5 illustrates LDA’s ability to isolate more specialised complaint categories. Right: Top-30 most salient terms across the corpus, computed from term frequency and cross-topic distinctiveness, highlighting influential words such as "equipment", "staff", "machines", and "membership". 

λ=1 provides the most frequent complaint terms overall, which is useful for understanding dominant themes. Lower λ values (λ=0.3) emphasises topic-specific terms, identifying unique characteristics of each theme (Fig. 7). Generally, λ=0.6 provides a good balance between frequency and distinctiveness and therefore offers optimal interpretable topic differentiation (Sievert and Shirley, 2014).
Topics 1, 2, and 4 formed the dominant complaint space (55.4% of tokens), representing equipment, staff, and facilities, the interconnected dimensions of operational failure (Fig. 6, 8).
Topics 3 and 5 form a separate but closely positioned cluster (combined 22.9%), representing membership and access issues (Fig. 6, 9).
Topic 7 contained non-English tokens and encoding artifacts that LDA failed to integrate, mirroring BERTopic's multilingual limitations (Fig. 10). 
Niche topics (8, 9, 10) captured amenities (5.0%), ambiance (3.1%), and operational timing (2.7%) (Fig. 11).
<img width="867" height="460" alt="image" src="https://github.com/user-attachments/assets/0cf9eb38-fea9-4d30-90b5-9c9594633101" />
Figure 7. Comparison of LDA topic term distribution for Topic 1 (23.2% of tokens) across two relevance scales (λ=1 and 0.3). Left: At λ=1.0, the ranking is dominated by absolute term frequency (“equipment”, “machines”, "people") and generic noise ("one", "get") is present; Right: As λ decreases to 0.3, the model prioritises terms that are exclusive to this topic, such as "dumbbells", "treadmills", and "bench". This progression confirms Topic 1 as a distinct cluster focused on gym equipment and overcrowding.

<img width="865" height="455" alt="image" src="https://github.com/user-attachments/assets/01cd4950-c4f1-47d4-9bb2-5c048f3442ef" />
<img width="432" height="454" alt="image" src="https://github.com/user-attachments/assets/938f728f-48dc-4b35-874d-6a3fd6c83b98" />


Figure 8. LDA topic term distribution for Topic 1, 2, and 4, collectively representing 55.4% of negative review tokens. With the relevance metric set at λ=0.6, these topics are dominated by equipment-related terms, staff complaints, and facility conditions.
<img width="863" height="460" alt="image" src="https://github.com/user-attachments/assets/ed347e95-2bb1-4863-96a7-6a3f5ab7a515" />

Figure 9. LDA topic term distribution for Topic 3 and 5, collectively representing 22.9% of negative review tokens. With the relevance metric set at λ=0.6, these topics are dominated by membership, transactional and access terms.

<img width="437" height="465" alt="image" src="https://github.com/user-attachments/assets/29c9acc4-9e62-439f-999b-a9f05f1be870" />

Figure 10. LDA topic term distribution for Topic 7, which appears isolated in the intertopic distance map, representing only 5.2% of negative review tokens. With the relevance metric set at λ=0.6, this topic is dominated by linguistic artefacts (single letters) and non-English terms such as “maskiner” (machines) and “udstyr” (equipment), highlighting LDA’s limitation in semantically integrating multilingual tokens with the wider topic space.

<img width="858" height="487" alt="image" src="https://github.com/user-attachments/assets/a6a7a0b1-d8d0-416c-abd6-dcd1414552c7" />
<img width="434" height="486" alt="image" src="https://github.com/user-attachments/assets/7e23ac0d-b44c-4b5b-a749-ea87cbe44362" />


Figure 11. LDA topic term distribution for 3 niche topics representing distinct, specialised complaints. With the relevance metric set at λ=0.6, Topic 8 (5.0% of tokens) focuses on amenity and infrastructure issues ("parking", "car", "aircon", "ventilation", "wifi", "hours", "open", "times"), capturing facility support services beyond core gym operations; Topic 9 (3.1% of tokens) isolates ambiance disruptions ("music", "loud", "classes", "locker", "instructors", "stolen", "lockers", "trainers"), representing environmental quality and security concerns; Topic 10 (2.7% of tokens) addresses operational timing problems ("overcrowded", "open", "closed", "hours", "opening", "pm"), highlighting overcrowding and access hour complaints.

Both approaches converged on core complaint themes (equipment, staff, facilities, and membership), validating the robustness of the findings. LDA’s 10 broad topics provided an interpretable strategic overview, while BERTopic’s fine-grained topics revealed specialised complaints that LDA integrated into larger categories. LDA’s lambda transparency supported understanding of topic composition, while BERTopic’s semantic modelling uncovered niche issues.

## Actionable Insights
Negative reviews were transformed into stakeholder-specific interventions addressing equipment quality, facility maintenance, hygiene standards, staff training, overcrowding management, and customer service improvements (Table 10).

<img width="755" height="717" alt="image" src="https://github.com/user-attachments/assets/b365f2b6-e2f9-4fc2-b45d-9a6cd33e7be2" />

Table 10. Key actionable insights derived from negative reviews, mapped to relevant stakeholders. Each insight outlines practical suggestions to address recurring gym issues.

## Conclusion
This analysis systematically extracted complaint patterns from 18507 negative gym reviews using multiple NLP approaches. Word frequency identified dominant terms; emotion analysis revealed anger triggers; BERTopic discovered granular subcategories; LDA validated classically; Phi4 enabled actionable translation. Convergence across all methods provides robust evidence of core pain points.

### The Complete Picture
Equipment problems are multifaceted and universal. Every method confirms equipment as the dominant complaint. Word frequency ranked "equipment/machines" highest. LDA's largest topic focused on equipment availability and overcrowding, with λ=0.3 revealing specific items: dumbbells, benches, treadmills, weight plates. BERTopic separated this into availability, broken/outdated machines, and peak-time inaccessibility.
Staff problems and unprofessionalism are confirmed by all methods as a serious risk for brand perception and reputation damage by provoking anger-driven reviews and causing general dissatisfaction.
Members face friction at multiple steps of membership and access. Streamlining these processes reduces complaints across multiple categories simultaneously.
Poor hygiene, facility maintenance, and amenities are important factors that may not cause immediate anger but will erode satisfaction slowly until members leave.
The 30 worst locations (17 in London) suffer compound failures. These sites require holistic operational audits, not isolated interventions.
Actionable insights from Phi4 have successfully suggested practical solutions relevant to all abovementioned matters.

### Limitations and Recommendations
Manual stopword removal left generic terms in word frequency distributions. While BERTopic and LDA's semantic modeling partially mitigated this by focusing on topic-distinctive terms, advanced preprocessing (domain-specific stopword lists, lemmatisation) would improve signal-to-noise ratios across all methods. 
Future works:
- Implementing multilingual embeddings and language-stratified analysis;
- Performing temporal analysis to track complaint evolution and seasonal patterns;
- Investigating location-specific niche complaints;

## References
Garg, J. (2025) What is PHI‐4? Understanding Microsoft’s 14 B Reasoning‐Focused LLM. Available at: https://www.gocodeo.com/post/what-is-phi-4-understanding-microsofts-14-b-reasoning-focused-llm?utm_source=chatgpt.com (Accessed: Jan 2026)
Krishnan, A. (2023) ‘Exploring the Power of Topic Modeling Techniques in Analyzing Customer Reviews: A Comparative Analysis’, arXiv, 19 August. https://doi.org/10.48550/arXiv.2308.11520
Mu, Y., Dong, C., Bontcheva, K., Song, X. (2024) ‘Large Language Models Offer an Alternative to the Traditional Approach of Topic Modelling.’, LREC-COLING-2024, doi:https://doi.org/10.48550/arxiv.2403.16248
Sievert, C. and Shirley, K. (2014). ‘LDAvis: A method for visualizing and interpreting topics.’ Proceedings of the Workshop on Interactive Language Learning, Visualization, and Interfaces, pages 63–70. doi:https://doi.org/10.3115/v1/W14-3110


_________________________________________________________________________________________________________
Word count (main text only, excluding cover, tables, figures, captions, and references): ~ 1400 words
