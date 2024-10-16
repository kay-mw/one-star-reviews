<TASK>

Analyze the attached data containing product reviews and provide evaluations in
JSON format.

<EVALUATION-CRITERIA>

Familiarise yourself with the below criteria, which you will use to generate an
overall quality rating ("score") ranging from 0-10.

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

<EVALUATION-EXAMPLES>

# Positive Reviews

### High Quality Example

```
{
  "rating": 5.0,
  "title": "A superb amateur telescope",
  "review_text": "This is not only the ideal scope for beginners but enough of
  an instrument to show you new things in the sky for years to come. It is sturdy
  and simple to operate. It has enough light-gathering power (more important than
  magnification) to reveal dim star clusters, nebulae, and galaxies and good
  enough optics to show you the surface of planets like Jupiter, Saturn, and
  Mars. Dobsonians like this one give by far the most view for the dollar, and
  the price on this one is great. Affordable as this scope is, there is no reason
  to give so much as a glance at the numerous poor quality 60mm refractors with
  exaggerated magnification claims. Some advice on selection. Dobsonians come in
  a range of sizes. A 114mm (4 ½') is a bit on the small side but still a fine
  instrument, especially if your ability to carry large objects is limited. A
  150-200mm (6-8') scope like this one is right in the middle of the recommended
  range. A 250mm (10') is on the big side, and you should buy one only if you are
  able-bodied. A very useful bonus is a full-sized 9x50mm finder scope. If
  competing models offer only a 6x30mm finder, you should factor in the cost of
  upgrading to the far better 50mm size. You will need eyepieces. Plossl-type
  eyepieces are good yet affordable: start with a 32mm and a 7 or 8mm.",
  "timestamp": 1170389642000,
  "product_title": "StarHopper 8"
}
```

### Low Quality Example

```
{
  "rating": 5.0,
  "title": "Great",
  "review_text": "Great",
  "timestamp": 1573310431862,
  "product_title": "Hypafix Retention Tape 2' X 10 Yard Roll Each"
}
```

# Negative Reviews

### High Quality Example

```
{
   "rating": 1.0,
   "title": "Unknown chemicals, no COA, no lots, no QC",
   "review_text": "I ordered Tianeptine sulfate from Nootropic source. The material
   is qualitatively different from what I've ordered from other sources. It’s a
   dense, free flowing powder instead of very light, fluffy powder. Second,
   Tianeptine sulfate is soluble in Dimethylsulfoxide at concentrations > 500mg/mL,
   with a faint purple color depending on the degree of hydration in the crystals.
   N.S. product had insolubles at 250mg/mL and was faint yellow. The melting point
   was very broad and significantly different from my reference compound indicating
   an impure, likely different compound. They don’t put lot numbers on their
   chemicals so there’s no way to trace it to a COA or any kind of internal
   testing. Very sloppy QC, I wouldn’t put something like this in my body.",
   "timestamp": 1588687728923,
   "product_title": "Tianeptine Sulfate Capsules"
}
```

### Low Quality Review

```
{
   "rating": 1.0,
   "title": "BAD!!",
   "review_text": "BAD!! SO BAD!! DO NOT BUY!!!",
   "timestamp": 1381841877000,
   "product_title": "Aioneus iPhone Charger Cable 2M, MFi Certified Lightning Cable
   Fast Charging iPhone Cable Lead Nylon Lightning to USB Cable for iPhone 14 13 12
   11 Pro Max XS XR X 8 7 6 Plus 5 SE"
}
```

<INSTRUCTIONS>

1. Analyze each review provided.
2. Use the evaluation criteria to assess the review.
3. Format your response as valid JSON following the output format.

<INPUT-DATA-SCHEMA>

The schema of the review data attached to this prompt is as follows:

- "rating": the star rating of the review, ranging from 1-5. (FLOAT)
- "review_title": the title of the review. (STRING)
- "review_text": the actual contents/text of the review. (STRING)
- "timestamp": the unique identifier of each review. (INTEGER)
- "product_title": the title of the product being reviewed. (STRING)

<REVIEW-DATA>
