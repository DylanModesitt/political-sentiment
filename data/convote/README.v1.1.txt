
======= Introduction ======= 

This README v1.1 (January 2008) for the v1.1 convote dataset comes
from the URL  http://www.cs.cornell.edu/home/llee/data/convote.html .
The only difference between this README (and the associated files) is
that a typo in the first line of graph_edge_data/edges_individual_document.v1.0.csv has been corrected.

======= Citation Info ======= 

This data was first used in Matt Thomas, Bo Pang, and Lillian Lee, 
"Get out the vote: Determining support or opposition from Congressional
floor-debate transcripts", Proceedings of EMNLP (2006).

@InProceedings{Thomas+Pang+Lee:06a,
  author =       {Matt Thomas and Bo Pang and Lillian Lee},
  title =        {Get out the vote: {Determining}  support or opposition from
  {Congressional} floor-debate transcripts},
  booktitle =    {Proceedings of EMNLP},
  pages={327--335},
  year =         2006
}

The original paper has been revised, with the updates including minor
changes to four of the reported data points.  An updated version is
available here:
  http://www.cs.cornell.edu/home/llee/papers/tpl-convote.home.html



======= Instructions and Contents ======= 

Please read the most up-to-date version of the paper, available at
  http://www.cs.cornell.edu/home/llee/papers/tpl-convote.home.html
as well as this README.  Important information, terminology,
motivations behind design decisions, and caveats are given in the
(update to) the EMNLP 2006 paper.

The rest of this file consists of two sections discussing the document
set, and then a section describing the data and policies that gives
the edge weights we derived.  These edge weights and related
statistics are provided in the data distribution, which allows for
experimental comparison with other graph-based document classifiers
upon the graphs we constructed, and for experimental comparison with
other methods for determining agreement between references or
documents.


======= Data description  ======= 

Our dataset includes three stages of tokenized speech-segment data,
corresponding to three different stages in the analysis pipeline we
employed.  The same speech segment may be represented in all three
stages.

- "data_stage_one" was used to identify by-name references to
  train our agreement classifier, which acts on such references.  All
  references in this dataset are annotated with a special set of
  characters of the form "xz1111111", where 1111111 is replaced by a
  seven-digit code indicating the House Member who we determined to be
  the target of the reference.  The first six digits of the code
  matches the index used to label the target Member's speech segments,
  (see description of our individual-file-naming convention, below).  The
  seventh digit is a relic from early experiments and was not used in
  our final study.

- "data_stage_two" was used to apply our agreement classifier to the
  test and development sets.  The only difference between
  data_stage_one and data_stage_two is that data_stage_two does not
  have any speech segments that contain the string "amendment" (or any
  superstring).  When we converted the results of our agreement
  classifier to graph-link weights, we did all normalization on a
  per-debate basis using the references mined from data_stage_two (the
  agreement classifier itself, however, was trained on all references
  found in the training set of data_stage_one, since this gave better
  performance on the development set than a data_stage_two-trained
  classifier).  The reference annotations mentioned for data_stage_one
  are also present in data_stage_two.

- "data_stage_three" is the dataset that was used for speech-segment
  support/oppose classification once the agreement classifier had
  been trained, validated, and applied to the test-set data.  It
  contains all the speeches in data_stage_two, except for
  single-sentence speeches containing the term "yield".

We stress again that in our construction of the graphs used to perform
speech-segment support/oppose classification, the nodes corresponded
to speech segments in data_stage_three, but the edges connecting
speech segments from different speakers corresponded to references
mined from data_stage_two.  As noted in the paper, we associated
references with pairs of speakers rather than pairs of speech segments
(the final graph link was drawn between an arbitrary speech segment by
one speaker and an arbitrary speech segment by the other), so we were
still able to use references from data_stage_two in the final graphs,
even in cases in which the speeches containing those references were
not present in data_stage_three.

Now, as for the speech-segment file-naming convention, 
###_@@@@@@_%%%%$$$_PMV is decoded as follows:

 - ### is an index identifying the bill under discussion in the
       speech segment (hence, this number also identifies the 'debate' to
       which the speech segment belongs)

 - @@@@@@ is an index identifying the speaker

 - %%%% is the index for the page of the Congressional record on which
   the speech segment appears, i.e., a number from 0001 to 3268 corresponding to one
   of the original HTML pages that we downloaded from govtrack.us .

 - $$$ is an index indicating the position of the speech segment within its
   page of the Congressional record.  Hence, for example, a file named
   055_400144_1031004_DON.txt would be the 4th speech on the 1031st
   HTML page of the record.

 - 'P' is replaced by a party indicator, D or R (or X if no
   corresponding party could be found).  As mentioned in the paper, we 
   purposely *did not* use this information in our experiments.

 - 'M' is replaced by an indicator of whether the bill under
   discussion is mentioned directly in the speech segment, or whether it is
   only referenced by another speech segment on the same page.  If the bill is
   directly mentioned in the current speech, the letter M appears in
   the file name; otherwise, the letter O appears.

 - 'V' is replaced by a vote indicator, Y or N, which serves as the
   ground-truth label for the speech.


======= Data Collection Procedure (for reference)  ======= 

We obtained our first data set (data_stage_one) via the following process:

 - We downloaded all available pages of the 2005 U.S. House record
 from govtrack.us.

 - For each page, we tallied the annotated references to each bill.
   The entire page was then associated with the bill receiving the
   most references (ties were broken in favor of the bill that reached
   the total number of references at an earlier point on the page).
   If a page could not be associated with a bill, it was discarded.

 - We downloaded all of govtrack.us's available XML files describing
   votes that took place on the House floor in 2005.  We then
   associated each of our Congressional record pages with the vote
   that was taken on the associated bill.  If the associated bill
   never came to a vote, the page was discarded.

 - Each page was parsed into speech segments, with a speech segment
   being any continuous utterance by a single member (HTML annotations
   on govtrack.us made it simple to identify such utterances, though
   some heuristics were needed to identify non-annotated speech
   breaks.)

 - Using our voting records, we associated each speech segment with a "yes" or
   "no" label according to the speaker's decision in the corresponding
   vote.  If the speaker abstained from the vote, the speech segment was
   discarded.

 - Each set of speeches corresponding to the same bill (and hence the
   same vote) was grouped into a "debate".  In order to limit our
   dataset to "interesting" debates, we kept only debates for which at
   least 20% of speeches were given the 'yes' label and at least 20%
   were given the 'no' label.


======= Data,  Policies, and Statistics for Graph Construction  ======= 

SVM scores and classification information needed to reconstruct the
graphs used in our (updated) EMNLP 2006 analysis can be found in the
following four files, which reside in the directory graph_edge_data/:

  edges_reference_set_full.v1.0.csv
  edges_reference_set_high_precision.v1.0.csv
  edges_individual_document.v1.0.csv
  edges_concatenated_document.v1.0.csv

and the calculations described in the procedures below are
implemented in the provided Excel spreadsheet, also residing in
the directory graph_edge_data :
  edge_calculations.v1.0.xls


The text below explains the procedure for setting up our graphs.
There are two parts: first, the basic procedure for setting up
individual document nodes, and second, the procedure for integrating
agreement information.  After each part of the procedure, we explain
how to interpret the values in the corresponding data files.

Part 1: Individual Document Modeling
------------------------------------

Procedure:
----------

- We use a trained SVM to assign an individual document score (representing a distance
  from the SVM's decision plane) to each speech in the development and test
  sets.  Positive scores correspond to "yes" classifications and negative scores
  correspond to "no" classifications.

- We normalize each document's score by dividing it by the standard deviation of
  all scores in the debate containing the document.

- For each debate, we build a graph with a source node, a sink node, and a node
  for each speech.  For each speech node, we add a directed edge from the source
  and a directed edge to the sink.  The strengths of these two edges always add
  up to 10000, and are determined as follows:
  - For a speech with a normalized score at or below -2, the edge from the
    source has strength 0 and the edge to the sink has strength 10000.
  - For a speech with a normalized score at or above +2, the edge from the
    source has strength 10000 and the edge to the sink has strength 0.
  - For a speech with normalized score between -2 and +2, we calculate the
    strength of the edge from the source as follows:

      strength_of_edge_from_source = (normalized score + 2) * 2500

    The strength of the edge to the sink can then be found by subtracting the
    above result from 10000.

 - Within each debate graph, all speeches by the same speaker are connected with
   links of effectively infinite strength (the actual strength value used is
   irrelevant as long as it is high enough to ensure that a minimum cut of the
   graph will never separate two speeches by the same speaker).

Data Files:
-----------

The files edges_individual_document and edges_concatenated_document
represent two classes of experiments, corresponding to two sources of
initial raw SVM scores.

In edges_individual_document, we use the SVM to assign a unique score
to each individual speech.  Results for experiments using this data
can be found in Table 4 in the paper.

When using the data in edges_concatenated_document, we continue to
have a node corresponding to each speech, but before calculating raw
SVM scores, we replace each speech with a concatenation of all of the
speaker's comments within the debate containing the speech.
Effectively, then, we use our graph to classify the concatenated texts
instead of the individual speeches, but we represent each concatenated
text with a set of n identical nodes, where n is the number of
speeches being concatenated (this use of multiple nodes to represent
the same concatenated-speech document allows us to obtain results that
are directly comparable to the results found when using the data in
edges_individual_document, so that the number of items classified in
each case stays the same).  Results for experiments using this data
can be found in Table 5 in the paper.

Each line in edges_individual_document and edges_concatenated_document
corresponds to a single speech, and is formatted as follows:

<speech filename>,<raw score>,<normalized score>,
  <strength of edge from source>,<strength of edge to sink>

The speech filename contains all information needed to identify the
speech, including a debate number, speaker id, true label, and unique
speech id, as described above.

Part 2: Agreement Modeling 
--------------------------

Procedure:
----------

- For each by-name reference to someone who made a speech in the
  debate in question according to the data_stage_two dataset, we use
  an SVM to obtain a raw agreement score.  Positive scores represent
  references that are classified as agreements, and negative scores
  represent references classified as disagreements. (The fact that we
  discarded references to House members who did not make a speech in
  the relevant debate means that there is not necessarily a one-to-one
  correspondence between the "xz" character sequences in data_stage_two
  and lines in the edges_reference_* files.)

- We normalize the SVM scores for each reference.  The normalization of scores
  within each debate depends on a parameter theta, which is described in the
  paper and below in "Data files" (where we discuss how different values of
  theta correspond to figures in our files).  The general normalization function
  is:

   normalized_score = (raw_score - theta) / (std. dev. of all reference
                                             scores in the debate)

- In our EMNLP 2006 paper (and its November 2006 update), we
  disregarded references with a negative normalized score.  For
  references with a positive normalized score, we convert the
  normalized score into an edge strength as follows:

   edge_strength = normalized_score * 2500 * alpha

  Within each experiment, we use the development-set debate graphs to
  find a value of alpha that maximizes accuracy, and we then apply
  this value in building the graphs for all debates in the test
  set.(In the case of three-way ties on the development set, we chose
  the intermediate value; for two-way ties, we broke the tie by
  choosing uniformly at random among the two choices.)

- For each reference, we use the calculated edge strength to produce a
  link between a speech by the speaker making the reference and a
  speech by the member being referenced.  Note the following:

    -- The particular speech nodes connected by the edge can be chosen
    arbitrarily, as long as they correspond to the appropriate
    speakers.  Since we chose to connect all speeches by the same
    speaker with links of virtually infinite strength, an edge from a
    speech by speaker A to a speech by speaker B will have the same
    effect on the graph's minimum cut, regardless of which particular
    speeches by speaker A and speaker B are chosen.

    -- In our experiments, since the graph modeling software we
    employed (Boris Cherkassky and Andrew Goldberg's PRF program) used
    directed edges, we created two edges for each agreement instance,
    one in each direction between a pair of nodes.

    -- Since our final set of references came from data_stage_two and
    our final set of speeches came from data_stage_three, we had some
    references for which the member making the reference and/or the
    member being referenced had no speech nodes in our graph.  Such
    references were discarded when we constructed our graphs, but they
    were included in the averages and standard deviations that we used
    to normalize our agreement scores.


- Finally, we take the minimum cut of each graph.  Speeches on the
  side of the source node are projected to have the "yes" label, and
  speeches on the side of the sink are projected to have the "no"
  label.

Data Files:
-----------

The files edges_reference_set_full and
edges_reference_set_high_precision contain reference data for two
types of experiments.

In edges_reference_set_full, we use theta = 0 when normalizing the
references' SVM scores.

In edges_reference_set_limited, we set theta equal to the average raw
SVM score for the references in a debate.  Thus, after normalizing the
scores, only the speeches with an above-average raw score have a
positive normalized score.  This is meant to raise the precision of
the set of agreements that make it into our graph model.

Results for our experiments with both values of theta can be found in
tables 4 and 5 in the paper.

Each line in each of these two files represents a single reference,
and can be interpreted as follows:

<true label>,<debate number>,<id of speaker making reference>,
  <id of speaker being referenced>,<raw score>,
  <normalized score>,<edge strength with c = 1>

Where <true label> is 1 for agreement or -1 for disagreement.

When building the graphs, we used the following values for the alpha
parameter mentioned above (these values were found to be optimal on the
development set):
 - When using data from edges_individual_document and
   edges_reference_set_full, alpha = 1.1
 - When using data from edges_individual_document and
   edges_reference_set_high_precision, alpha = 5
 - When using data from edges_concatenated_document and
   edges_reference_set_full, alpha = 1.7
 - When using data from edges_concatenated_document and
   edges_reference_set_high_precision, alpha = 5
