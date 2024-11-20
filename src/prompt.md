# Task

1. Analyze each review provided.
2. Use the evaluation criteria to assess the review.
3. Format your response as valid JSON following the output format.

# Evaluation Criteria

Familiarise yourself with the below criteria, which you will use to generate an
overall quality evaluation ("evaluation"), which can be any integer greater than
or equal to 0 and less than or equal to 10.

Reviews should be evaluated based on:

1. How detailed and useful the information is.

   - A perfect review will give highly detailed information. For example, they
     will not just say "This product works great", but will elaborate on why
     said product works great.
   - A perfect review will tend to give contextual information. For example,
     "This product worked great for me, but if you have x characteristic it may
     not work well".
   - A perfect review will be of a reasonable length. This means striking a
     balance between sufficient detail, whilst avoiding the inclusion of
     unimportant information.

2. How objective the review is.

   - A perfect review will make very few personal "I" or "my" statements.
   - A perfect review will stick to objective language, using emotive adjectives
     only when warranted. They may say "The manufacturing was sloppy", but would
     not say "The manufacturing was complete garbage".

3. How relevant the review is.

   - A perfect review will be highly relevant to the product. A great test of
     relevance is: could you copy the review to a completely different product
     and have it still make sense? If the answer is no, the review is highly
     relevant.

# Input Data Schema

The schema of the review data attached to this prompt is as follows:

- `<review_title: string>`: the title of the review.
- `<review_text: string>`: the actual contents/text of the review.
- `<timestamp: int>`: the unique identifier of each review.
- `<product_title: string>`: the title of the product being reviewed.

# Output Schema

The schema of your output should be as follows:

`[{"timestamp": <timestamp: int>, "evaluation": <evaluation: int>}]`

Where `<timestamp: int>` is the timestamp for that review (from the input data),
and `<evaluation: int>` is your evaluation of that review based on the
previously described criteria.

# Review Data
