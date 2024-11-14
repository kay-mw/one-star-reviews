<TASK>

Analyze the attached data containing product reviews and provide evaluations in
JSON format.

<EVALUATION-CRITERIA>

Familiarise yourself with the below criteria, which you will use to generate an
overall quality evaluation ("score"), which can be any integer greater than or
equal to 0 and less than or equal to 10.

Reviews should be evaluated based on:

1. How detailed and useful the information is.

   - A perfect review will give highly detailed information, e.g. not just
     saying "it works great", but saying why it works great.
   - A perfect review will tend to give contextual information, e.g. "it worked
     great for me, but if you have x characteristic it may not work well"
   - A perfect review will be of a reasonable length. There is no specific
     character limit here, it is about striking a balance between sufficient
     detail whilst avoiding the inclusion of unimportant information.

2. How objective the review is.

   - A perfect review will make very few personal "I" or "my" statements.
   - A perfect review will stick to objective language, using emotive adjectives
     only when warranted. They may say "the manufacturing was sloppy", but would
     not say "the manufacturing was complete garbage".

3. How relevant the review is.

   - A perfect review will be highly relevant to the product, meaning you could
     not copy the review to a completely different product and have it still
     make sense.

<INSTRUCTIONS>

1. Analyze each review provided.
2. Use the evaluation criteria to assess the review.
3. Format your response as valid JSON following the output format.

<INPUT-DATA-SCHEMA>

The schema of the review data attached to this prompt is as follows:

- `<review_title: string>`: the title of the review.
- `<review_text: string>`: the actual contents/text of the review.
- `<timestamp: int>`: the unique identifier of each review.
- `<rating: float>`: the star rating the reviewer gave this product, ranging
  from 1-5.
- `<product_title: string>`: the title of the product being reviewed.

<OUTPUT-SCHEMA>

The schema of your output should be as follows:

`[{"timestamp": <timestamp: int>, "score": <score: int>}]`

Where `<timestamp: int>` is the timestamp for that review, from the input data,
and `<score: int>` your evaluation of that review based on the previously
described criteria.

IMPORTANT NOTE: the `score` field in your output is distinct from the `rating`
field in the input data. Remember, the score is your _evaluation_ of the review,
based on the review assessment criteria - this is a value _you_ generate. In
contrast, the `rating` field just represents the star rating that reviewer gave
a particular product.

<REVIEW-DATA>
