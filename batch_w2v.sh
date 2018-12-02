#!/bin/bash
python WordEmbeddings_Cache.py --embdir "/shared/preprocessed/qning2/MagnitudeEmbeddings" --embname "wiki-news-300d-1M-light.magnitude"
python WordEmbeddings_Cache.py --embdir "/shared/preprocessed/qning2/MagnitudeEmbeddings" --embname "wiki-news-300d-1M-medium.magnitude"
python WordEmbeddings_Cache.py --embdir "/shared/preprocessed/qning2/MagnitudeEmbeddings" --embname "GoogleNews-vectors-negative300-medium.magnitude"
python WordEmbeddings_Cache.py --embdir "/shared/preprocessed/qning2/MagnitudeEmbeddings" --embname "GoogleNews-vectors-negative300-light.magnitude"

