#!/usr/bin/env python
# coding: utf-8

# # Candidate Generation
# 
# > for a given prefix we extract the end-term as shown in Figure 1. We match all the suffixes that
# > start with the end-term from our precomputed set. These selected suffixes are appended to the
# > prefix to generate synthetic suggestion candidates. For example, the prefix "cheap flights fro"
# > is matched with the suffix "from seattle" to generate the candidate "cheap flights from seattle".
# > Note that many of these synthetic suggestion candidates are likely to not have been observed by
# > the search engine before. We merge these synthetic suggestions with the set of candidates
# > selected from the list of historically popular queries. This combined set of candidates is used
# > for ranking as we will describe in Sec 4.

# In[ ]:


import pandas as pd


# In[ ]:


def end_term(query):
    """
    end_term('cheapest flight fro') = 'fro'
    end_term('cheapest flight from') = 'from'
    end_term('cheapest flight from ') = 'from '
    end_term('cheapest flight from n') = 'n'
    """
    if query.endswith(' '):
        return query[query[:-1].rfind(' ')+1:]
    else:
        return query[query.rfind(' ')+1:]

