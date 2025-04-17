import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ClusterInfo:
    cluster_id: int
    papers: List[str]  # List of DOIs

@dataclass
class TimelinePoint:
    date: datetime
    papers: List[str]  # List of DOIs
    cluster_id: int

class PaperClusterAnalyzer:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def cluster_papers_with_scores(self, papers: List[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], List[TimelinePoint]]:
        """
        Clusters papers based on composite scores and organizes them into timeline points.

        Args:
            papers (List[Dict[str, Any]]): List of rated papers with composite scores.

        Returns:
            Tuple[Dict[int, Dict[str, Any]], List[TimelinePoint]]: Clusters as a dictionary keyed by cluster_id and timeline points.
        """
        # Example clustering based on composite_score thresholds
        clusters = {}
        timeline = []

        for paper in papers:
            score = paper.get('composite_score', 0)
            doi = paper['doi']
            # Define clusters based on score ranges
            if score >= 80:
                cluster_id = 1  # High relevance
            elif score >= 60:
                cluster_id = 2  # Medium relevance
            else:
                cluster_id = 3  # Low relevance

            if cluster_id not in clusters:
                clusters[cluster_id] = ClusterInfo(cluster_id=cluster_id, papers=[])
            clusters[cluster_id].papers.append(doi)

        # Convert ClusterInfo instances to dictionaries and create a dictionary keyed by cluster_id
        cluster_infos = {cluster.cluster_id: asdict(cluster) for cluster in clusters.values()}

        # Organize timeline points based on formatted_date
        for paper in papers:
            date_str = paper.get('formatted_date', '1970-01-01')
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                date_obj = datetime(1970, 1, 1)
            # Find the cluster this paper belongs to
            cluster_id = next((c_id for c_id, c in cluster_infos.items() if paper['doi'] in c['papers']), 0)
            timeline.append(TimelinePoint(date=date_obj, papers=[paper['doi']], cluster_id=cluster_id))

        # Sort timeline by date
        timeline_sorted = sorted(timeline, key=lambda x: x.date)

        return cluster_infos, timeline_sorted
