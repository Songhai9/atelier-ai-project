# Atelier AI — Data Card

## Status

Source documentation reviewed.

Data acquisition, local inspection and validation are pending.
No local row counts, date boundaries or data-quality findings are
reported at this stage.

## Source

- Dataset: H&M Personalized Fashion Recommendations
- Publisher: H&M Group
- Platform: Kaggle
- Documentation review date: 2026-10-22

The source benchmark predicts purchases during the seven days following
the supplied purchase history.

Atelier AI will define historical development and test periods within
the available transaction history for local evaluation.

## Initial File Scope

| File | V1 decision | Intended project use |
|---|---|---|
| transactions_train.csv | Required | Purchase histories and evaluation targets |
| articles.csv | Required | Product attributes and candidate feature exploration |
| customers.csv | Required | Customer population and metadata inspection |
| sample_submission.csv | Optional | Kaggle submission-format reference |
| images/ | Deferred | Potential future visual or multimodal experiments |

Including a file in the inspection scope does not imply that every
column will be used for modelling.

## Expected Relationships

Transactions are expected to reference customers through `customer_id`
and articles through `article_id`.

Key uniqueness, identifier compatibility and referential integrity will
be checked after acquisition.

Identifiers will be handled as identifiers rather than numerical
measurements, preserving their source representation.

## Initial Handling Decisions

- Preserve the acquired source files without modification.
- Retain repeated transaction rows during acquisition.
- Investigate repetition before applying any deduplication rule.
- Distinguish transaction history from distinct-product relevance
  labels used for ranking evaluation.
- Record acquisition dates, file sizes and SHA-256 checksums.
- Keep source data outside Git.

## Temporal Availability

Transaction-derived features and candidates must use information
available before each prediction cutoff.

Customer and product metadata will be reviewed for historical
availability. Their presence in the supplied files does not establish
that every attribute was known at every historical prediction date.

Time-varying attributes without sufficient temporal provenance will
not be assumed safe for historical modelling.

## Access and Publication

Access is subject to the competition rules.

The rules describe non-commercial, academic and educational uses and
restrict redistribution of competition data.

Source files will not be committed to the repository. Notebook outputs
and other published artifacts will be reviewed to avoid reproducing
restricted records or images.

## Interpretation Limits

Observed purchases are incomplete signals of customer preferences.

The project does not assume that historical relevance establishes
current stock availability or that a ranking score is a calibrated
purchase probability.

Results on this dataset will not establish performance in a luxury
retail setting or a measurable commercial benefit.

## Pending Validation

- Confirm acquired filenames, sizes and checksums.
- Inspect schemas, types and row counts.
- Establish the meaning and granularity of records.
- Check missing values, repeated records and key relationships.
- Inspect temporal coverage and define evaluation boundaries.
- Document feature-availability assumptions and unresolved limitations.

## References

- [Dataset documentation](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)
- [Competition rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules)
- [Project brief](project_brief.md)
- [Evaluation strategy](evaluation.md)