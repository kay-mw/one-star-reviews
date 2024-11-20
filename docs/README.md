# One Star Reviews

Should you trust negative reviews more than positive ones? Some people think
[one-star reviews](https://near.blog/read-the-one-star-reviews/) are more
valuable, arguing that the frustration of a defective product motivates more
detailed feedback than satisfaction with a good one. But is this actually true?
This project uses machine learning and statistical analysis to investigate this
question empirically.

# The Process

![flow-chart](../media/flow-chart.svg)

The analysis pipeline consisted of four main stages:

1. **Data Collection**: Used the
   [2023 Amazon Reviews Dataset](https://amazon-reviews-2023.github.io/)
   containing ~100GB of review and product data.

2. **Data Processing**: Implemented dual processing approaches:

   - Spark for distributed computing capabilities
   - Polars for efficient single-node operations

3. **Review Quality Assessment**:

   - Manually labeled 1,020 reviews (102 sets of 10) to create training data
   - Evaluated reviews on three criteria:

     - Information detail and usefulness
     - Objectivity and lack of emotional bias
     - Product-specific relevance

   - Fine-tuned Gemini 1.5 Flash on this labeled dataset
   - Validated model outputs through manual inspection of score distributions
     and edge cases

4. **Statistical Analysis**: Analyzed relationships between review quality and
   various factors using a sample of 60k+ reviews.

# Key Findings

## Review Quality Distribution

The first striking observation was that high-quality reviews are rare. Our
analysis shows that approximately 90% of reviews score below 5/10 on our quality
metric.

![pct-eval](../media/pct-eval.png)

Common low-quality patterns include:

- Single-line reactions ("Great product!!")
- Non-informative statements ("As expected")
- Pre-purchase comments ("Just ordered, can't wait!")

The average review contains just 221 characters (±429) or roughly 44 words. Even
longer reviews often focus on emotional reactions rather than objective
analysis.

## Temporal Validation

To validate our dataset's representativeness, we compared review volume over
time with Amazon's stock performance:

<div style="display: flex; flex-wrap: wrap;">
   <div style="vertical-align: middle; width: 100%;">
      <img style="vertical-align: middle; width: 49%;" src="../media/running-total.png" />
      <img style="vertical-align: middle; width: 49%;" src="../media/amazon-stock.png" />
   </div>
</div>

The similar growth patterns (particularly in the mid-2010s) suggest our dataset
accurately captures Amazon's historical review trends.

## Main Hypothesis Results

Our analysis confirms that negative reviews tend to be higher quality, but with
some nuances:

![avg-eval-by-rating](../media/avg_eval_by_rating.png)

Key findings:

- Weak-to-moderate negative correlation (r = -0.27) between star rating and
  review quality
- 2-star reviews actually scored higher than 1-star reviews
- 3-star reviews nearly matched 1-star reviews in quality
- The data suggests reading all reviews ≤3 stars rather than just 1-star reviews

## Additional Insights

### Verified Purchase Status

Unexpectedly, non-verified purchase reviews scored higher in quality (r =
-0.22):

![avg-eval-by-purchase](../media/avg_eval_by_purchase.png)

This could indicate review manipulation, with companies potentially gaming the
verified purchase system for positive reviews.

### Helpful Votes

Review quality showed the strongest correlation with helpful votes:

![avg-eval-by-help](../media/avg_eval_by_help.png)

Note: The sample size decreases significantly for reviews with >300 votes, so
these results should be interpreted cautiously.

### Price Impact

Higher-priced items received better quality reviews:

![avg-eval-by-price](../media/avg_eval_by_price.png)

This aligns with the hypothesis that greater financial investment motivates more
thorough reviewing.

# Practical Recommendations

When reading Amazon reviews, our analysis suggests:

1. **Sort by "Top reviews"** rather than "Most recent"
   - The "helpful votes" correlation suggests this surfaces better content
2. **Include non-verified purchase reviews**
   - Don't limit yourself to verified purchases, as they don't correlate with
     higher quality
3. **Focus on ≤3 star reviews**
   - These consistently contain more detailed, objective information
4. **Consider product price context**
   - Expect more detailed reviews on higher-priced items
5. **Use format filters selectively**
   - While not directly analyzed, these can help focus on relevant variations of
     products

# Future Work

Planned improvements include:

- Analysis of review quality variations across product categories
- Investigation of seasonal review quality patterns
- Expansion of the training dataset to improve model calibration

# Technical Details

The complete codebase and detailed technical documentation are available in the
repository. Key implementation notes:

- Review quality model training code and evaluation criteria
- Data processing pipelines for both Spark and Polars
- Statistical analysis script and SQL queries
- Visualization generation scripts
