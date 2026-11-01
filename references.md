# Pre-vetted references (IEEE format)

Each entry below was checked on 2026-10-29: the "Available:" URL was opened and shows the work, and the
bibliographic fields were copied from the publisher / dblp record (page ranges are given only where that record showed them). **Before submission, Maksim opens
every URL he cites once more** (Gate 0: a dead link = 0 points). Cite only what the report actually uses
(3–7 entries); do not add references that were not checked this way. DOIs given in brackets are the
canonical identifiers; they were taken from the dblp/publisher record, not opened — open them too if cited.

[1] Y. Hu, Y. Koren, and C. Volinsky, "Collaborative filtering for implicit feedback datasets," in *Proc. 8th IEEE Int. Conf. Data Mining (ICDM)*, Pisa, Italy, 2008, pp. 263–272, doi: 10.1109/ICDM.2008.22. [Online]. Available: https://dblp1.uni-trier.de/rec/bibtex/conf/icdm/HuKV08. [Accessed: 2026-10-29].
    - use for: implicit-feedback ALS (confidence weighting of watch time), why ratings-style MF is wrong here.
    - note: dblp sometimes shows a bot-check page; the IEEE Xplore page via the DOI is the primary source.

[2] R. Agrawal and R. Srikant, "Fast algorithms for mining association rules in large databases," in *Proc. 20th Int. Conf. Very Large Data Bases (VLDB)*, Santiago de Chile, 1994, pp. 487–499. [Online]. Available: https://vldb.org/dblp/db/conf/vldb/vldb94-487.html. [Accessed: 2026-10-29].
    - PDF mirror also checked: https://www.cs.columbia.edu/~gravano/Qual/Papers/agrawal94.pdf
    - use for: Apriori, support/confidence, downward closure.

[3] S. Brin and L. Page, "The anatomy of a large-scale hypertextual Web search engine," *Computer Networks and ISDN Systems*, vol. 30, no. 1–7, pp. 107–117, 1998, doi: 10.1016/S0169-7552(98)00110-X. [Online]. Available: https://www2.math.upenn.edu/~kazdan/210/LectureNotes/Google/Brin-Page.html. [Accessed: 2026-10-29].
    - use for: PageRank definition (§2.1 of the paper), damping factor.

[4] T. H. Haveliwala, "Topic-sensitive PageRank," in *Proc. 11th Int. World Wide Web Conf. (WWW 2002)*, Honolulu, HI, USA, 2002, doi: 10.1145/511446.511513. [Online]. Available: https://archives.iw3c2.org/www2002/CDROM/refereed/127.html. [Accessed: 2026-10-29].
    - use for: personalized restart vector (PPR), i.e. the user-specific teleport distribution.

[5] P. Cremonesi, Y. Koren, and R. Turrin, "Performance of recommender algorithms on top-N recommendation tasks," in *Proc. 4th ACM Conf. Recommender Systems (RecSys)*, Barcelona, Spain, 2010, pp. 39–46, doi: 10.1145/1864708.1864721. [Online]. Available: https://bibtex.github.io/RecSys-2010-CremonesiKT.html. [Accessed: 2026-10-29].
    - use for: top-N evaluation protocol, popularity as a strong baseline, popularity bias of accuracy metrics.

[6] P. Lops, M. de Gemmis, and G. Semeraro, "Content-based recommender systems: State of the art and trends," in *Recommender Systems Handbook*, F. Ricci, L. Rokach, B. Shapira, and P. B. Kantor, Eds. Boston, MA, USA: Springer, 2011, doi: 10.1007/978-0-387-85820-3_3. [Online]. Available: https://www.ise.bgu.ac.il/faculty/liorr/recsyshb/chContent.pdf. [Accessed: 2026-10-29].
    - use for: CBF (TF-IDF item profiles, user profile as weighted sum), its over-specialisation limitation.

[7] R. Burke, "Hybrid recommender systems: Survey and experiments," *User Modeling and User-Adapted Interaction*, vol. 12, no. 4, pp. 331–370, 2002, doi: 10.1023/A:1021240730564. [Online]. Available: https://vivo.colorado.edu/display/pubid_285008. [Accessed: 2026-10-29].
    - use for: hybridisation types (weighted, switching, feature-augmentation) — the hybrid here is feature augmentation + weighted.

[8] P. Covington, J. Adams, and E. Sargin, "Deep neural networks for YouTube recommendations," in *Proc. 10th ACM Conf. Recommender Systems (RecSys)*, Boston, MA, USA, 2016, doi: 10.1145/2959100.2959190. [Online]. Available: https://research.google/pubs/pub45530. [Accessed: 2026-10-29].
    - use for: two-stage architecture (candidate generation → ranking), which the hybrid copies.

[9] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, Long Beach, CA, USA, 2017. [Online]. Available: https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf. [Accessed: 2026-10-29].
    - use for: the re-ranker.

[10] B. Hamner et al., "Metrics: Machine learning evaluation metrics (ml_metrics), `average_precision.py`," GitHub repository. [Online]. Available: https://github.com/benhamner/Metrics. [Accessed: 2026-10-29].
    - use for: the MAP@K definition used on Kaggle (denominator min(|actual|, k)).

[11] B. Frederickson, "implicit: Fast Python collaborative filtering for implicit datasets," GitHub repository. [Online]. Available: https://github.com/benfred/implicit. [Accessed: 2026-10-29].
    - cite only if the `implicit` library is actually used (otherwise cite the NumPy ALS as own implementation of [1]).

[12] O. Gurbych, "sweet.tv – TV program recommender," Kaggle community competition, 2020. [Online]. Available: https://www.kaggle.com/competitions/sweettv-tv-program-recommender. [Accessed: TODO-MAKSIM date].
    - the data source and the label rule (Data tab). Maksim opens it himself and fills the access date.

Not vetted (do not cite unless Maksim checks them himself): BPR (Rendle et al., UAI 2009, arXiv:1205.2618); Ferrari Dacrema et al., RecSys 2019 ("Are we really making much progress?").
