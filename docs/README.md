# Unveiling the Nuances of Amazon Reviews

Amazon reviews are a ubiquitous feature of online shopping, guiding purchasing
decisions for millions of consumers. However, the reliability and quality of
these reviews can vary significantly. In navigating this jungle of product
evaluations, [some people](https://near.blog/read-the-one-star-reviews/) have
noticed a trend: more negative reviews tend to be higher quality.

But is this actually true? To find out, I created this project – a deep-dive
into a dataset of Amazon reviews, aiming to answer this question, uncover hidden
patterns and insights, and hopefully equip consumers with the knowledge to
navigate product feedback effectively along the way.

# The Process

![flow-chart](../media/flow-chart.svg)

To conduct this analysis, a large dataset of Amazon reviews from the
[McAuley Lab](https://amazon-reviews-2023.github.io/main.html) were collected
and preprocessed. This involved handling large datasets (up to 30GB each),
complex data types (Map/Struct), and perfoming speed/memory optimizations in
PySpark/Polars. Each review in the refined dataset were then given quality
scores using a fine-tuned Large Language Model (LLM). This final dataset was
then analyzed using correlation, regression, and visualizations to identify key
trends and relationships.

# Key Findings

- **Review Quality Distribution:** The analysis revealed a skewed distribution
  of review quality, with a majority of reviews exhibiting average quality. This
  highlights the importance of discerning high-quality reviews from the rest.

![pct-eval](../media/pct-eval.png)

- **Impact of Star Ratings:** A moderate correlation was observed between star
  ratings and review quality. This supports the idea that more negative reviews
  (3\* and below) tend to be superior to positive reviews (4\* and above).

![avg-eval-by-rating](../media/avg_eval_by_rating.png)

- **Verified Purchases:** Surpisingly, reviews marked as "Verified purchases"
  were on average lower quality than "unverified" reviews. This contradicts the
  common assumption that unverified reviews are lower quality, given they can be
  easily botted or faked.

![avg-eval-by-purchase](../media/avg_eval_by_purchase.png)

- **Helpful Votes:** In a more expected turn, the number of helpful votes a
  review receives proved to be a strong indicator of its quality. Reviews with
  more helpful votes were generally more informative and reliable.

![avg-eval-by-help](../media/avg_eval_by_help.png)

- **Price:** A weak correlation was found between product price and review
  quality. While higher-priced items may attract more detailed reviews, price
  alone is not a guarantee of review quality.

![avg-eval-by-price](../media/avg_eval_by_price.png)

# Practical Implications

These findings suggest that, when reading reviews, consumers should prioritize:

- Negative/critical reviews (3\* and below).
- Reviews with more helpful votes (especially those with 300+).
- Reviews that **aren't** verified purchases.
- Reviews for more expensive products.

# Conclusion

This project provides a data-driven perspective on the intricacies of Amazon
reviews, empowering consumers to make informed purchasing decisions. By
understanding the factors that influence review quality, consumers can
effectively leverage product feedback to navigate the vast landscape of online
shopping.
