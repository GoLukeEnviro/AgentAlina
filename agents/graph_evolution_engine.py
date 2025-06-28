#!/usr/bin/env python3
"""
Graph Evolution Engine

Selbstentwickelnde Knowledge Graph Engine für dynamische Schema-Evolution
und automatische Beziehungsoptimierung.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from config.env_config import get_neo4j_config
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NodePattern:
    """Muster für Knoten im Graph."""
    label: str
    properties: Dict[str, Any]
    frequency: int
    relationships: List[str]
    cluster_id: Optional[int] = None

@dataclass
class RelationshipPattern:
    """Muster für Beziehungen im Graph."""
    type: str
    source_label: str
    target_label: str
    properties: Dict[str, Any]
    frequency: int
    strength: float

@dataclass
class SchemaEvolution:
    """Schema-Evolution Event."""
    timestamp: datetime
    evolution_type: str  # 'node_merge', 'relationship_creation', 'property_optimization'
    description: str
    affected_nodes: List[str]
    affected_relationships: List[str]
    performance_impact: float

class GraphEvolutionEngine:
    """Engine für selbstentwickelnde Knowledge Graphs."""
    
    def __init__(self):
        self.neo4j_config = get_neo4j_config()
        self.driver = None
        self.node_patterns: List[NodePattern] = []
        self.relationship_patterns: List[RelationshipPattern] = []
        self.evolution_history: List[SchemaEvolution] = []
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.evolution_running = False
        
    async def initialize(self):
        """Initialisiert die Graph Evolution Engine."""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_config['uri'],
                auth=(self.neo4j_config['username'], self.neo4j_config['password'])
            )
            await self._analyze_current_schema()
            await self._load_evolution_history()
            logger.info("Graph Evolution Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Graph Evolution Engine: {e}")
            raise
    
    async def close(self):
        """Schließt Verbindungen."""
        if self.driver:
            self.driver.close()
    
    async def _analyze_current_schema(self):
        """Analysiert das aktuelle Graph-Schema."""
        # Analysiere Knoten-Muster
        await self._discover_node_patterns()
        
        # Analysiere Beziehungs-Muster
        await self._discover_relationship_patterns()
        
        # Clustere ähnliche Knoten
        await self._cluster_similar_nodes()
        
        logger.info(f"Discovered {len(self.node_patterns)} node patterns and "
                   f"{len(self.relationship_patterns)} relationship patterns")
    
    async def _discover_node_patterns(self):
        """Entdeckt Muster in Knoten."""
        query = """
        MATCH (n)
        WITH labels(n) as node_labels, keys(n) as node_properties, count(*) as frequency
        WHERE frequency > 1
        RETURN node_labels, node_properties, frequency
        ORDER BY frequency DESC
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                labels = record['node_labels']
                properties = record['node_properties']
                frequency = record['frequency']
                
                if labels:  # Nur Knoten mit Labels
                    # Hole Beziehungstypen für diese Knoten
                    rel_query = """
                    MATCH (n)-[r]-()
                    WHERE $label IN labels(n)
                    RETURN DISTINCT type(r) as rel_type
                    """
                    
                    relationships = []
                    rel_result = session.run(rel_query, {'label': labels[0]})
                    for rel_record in rel_result:
                        relationships.append(rel_record['rel_type'])
                    
                    pattern = NodePattern(
                        label=labels[0] if labels else 'Unknown',
                        properties={prop: 'dynamic' for prop in properties},
                        frequency=frequency,
                        relationships=relationships
                    )
                    self.node_patterns.append(pattern)
    
    async def _discover_relationship_patterns(self):
        """Entdeckt Muster in Beziehungen."""
        query = """
        MATCH (a)-[r]->(b)
        WITH labels(a)[0] as source_label, type(r) as rel_type, labels(b)[0] as target_label,
             keys(r) as rel_properties, count(*) as frequency
        WHERE frequency > 1 AND source_label IS NOT NULL AND target_label IS NOT NULL
        RETURN source_label, rel_type, target_label, rel_properties, frequency
        ORDER BY frequency DESC
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                source_label = record['source_label']
                rel_type = record['rel_type']
                target_label = record['target_label']
                properties = record['rel_properties']
                frequency = record['frequency']
                
                # Berechne Beziehungsstärke basierend auf Häufigkeit
                strength = min(1.0, frequency / 100.0)  # Normalisiert auf 0-1
                
                pattern = RelationshipPattern(
                    type=rel_type,
                    source_label=source_label,
                    target_label=target_label,
                    properties={prop: 'dynamic' for prop in properties},
                    frequency=frequency,
                    strength=strength
                )
                self.relationship_patterns.append(pattern)
    
    async def _cluster_similar_nodes(self):
        """Clustert ähnliche Knoten für potenzielle Zusammenführung."""
        if len(self.node_patterns) < 2:
            return
        
        # Erstelle Feature-Vektoren für Knoten-Muster
        features = []
        for pattern in self.node_patterns:
            # Kombiniere Label, Properties und Relationships zu Text
            text_features = [
                pattern.label,
                ' '.join(pattern.properties.keys()),
                ' '.join(pattern.relationships)
            ]
            features.append(' '.join(text_features))
        
        try:
            # TF-IDF Vektorisierung
            tfidf_matrix = self.vectorizer.fit_transform(features)
            
            # DBSCAN Clustering
            clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
            cluster_labels = clustering.fit_predict(tfidf_matrix.toarray())
            
            # Weise Cluster-IDs zu
            for i, pattern in enumerate(self.node_patterns):
                pattern.cluster_id = int(cluster_labels[i]) if cluster_labels[i] != -1 else None
            
            # Protokolliere Cluster
            clusters = defaultdict(list)
            for pattern in self.node_patterns:
                if pattern.cluster_id is not None:
                    clusters[pattern.cluster_id].append(pattern.label)
            
            for cluster_id, labels in clusters.items():
                if len(labels) > 1:
                    logger.info(f"Cluster {cluster_id}: {labels}")
                    
        except Exception as e:
            logger.warning(f"Failed to cluster nodes: {e}")
    
    async def _load_evolution_history(self):
        """Lädt Evolution-Historie."""
        query = """
        MATCH (e:SchemaEvolution)
        RETURN e
        ORDER BY e.timestamp DESC
        LIMIT 100
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                evolution_data = record['e']
                evolution = SchemaEvolution(
                    timestamp=datetime.fromisoformat(evolution_data['timestamp']),
                    evolution_type=evolution_data['evolution_type'],
                    description=evolution_data['description'],
                    affected_nodes=json.loads(evolution_data.get('affected_nodes', '[]')),
                    affected_relationships=json.loads(evolution_data.get('affected_relationships', '[]')),
                    performance_impact=evolution_data.get('performance_impact', 0.0)
                )
                self.evolution_history.append(evolution)
        
        logger.info(f"Loaded {len(self.evolution_history)} evolution events")
    
    async def evolve_schema(self) -> List[SchemaEvolution]:
        """Führt Schema-Evolution durch.
        
        Returns:
            Liste der durchgeführten Evolutionen
        """
        if self.evolution_running:
            return []
        
        try:
            self.evolution_running = True
            evolutions = []
            
            # 1. Merge ähnliche Knoten
            merge_evolutions = await self._merge_similar_nodes()
            evolutions.extend(merge_evolutions)
            
            # 2. Optimiere Beziehungen
            relationship_evolutions = await self._optimize_relationships()
            evolutions.extend(relationship_evolutions)
            
            # 3. Erstelle neue Beziehungen basierend auf Mustern
            new_rel_evolutions = await self._create_inferred_relationships()
            evolutions.extend(new_rel_evolutions)
            
            # 4. Optimiere Properties
            property_evolutions = await self._optimize_properties()
            evolutions.extend(property_evolutions)
            
            # Speichere Evolutionen
            for evolution in evolutions:
                await self._store_evolution(evolution)
                self.evolution_history.append(evolution)
            
            logger.info(f"Completed schema evolution with {len(evolutions)} changes")
            return evolutions
            
        except Exception as e:
            logger.error(f"Failed to evolve schema: {e}")
            return []
        finally:
            self.evolution_running = False
    
    async def _merge_similar_nodes(self) -> List[SchemaEvolution]:
        """Führt ähnliche Knoten zusammen.
        
        Returns:
            Liste der Merge-Evolutionen
        """
        evolutions = []
        
        # Gruppiere Knoten nach Cluster
        clusters = defaultdict(list)
        for pattern in self.node_patterns:
            if pattern.cluster_id is not None:
                clusters[pattern.cluster_id].append(pattern)
        
        for cluster_id, patterns in clusters.items():
            if len(patterns) < 2:
                continue
            
            # Prüfe ob Merge sinnvoll ist
            if await self._should_merge_nodes(patterns):
                merge_evolution = await self._perform_node_merge(patterns)
                if merge_evolution:
                    evolutions.append(merge_evolution)
        
        return evolutions
    
    async def _should_merge_nodes(self, patterns: List[NodePattern]) -> bool:
        """Prüft ob Knoten zusammengeführt werden sollten.
        
        Args:
            patterns: Liste von Knoten-Mustern
            
        Returns:
            True wenn Merge empfohlen wird
        """
        # Prüfe Ähnlichkeit der Properties
        all_properties = set()
        for pattern in patterns:
            all_properties.update(pattern.properties.keys())
        
        # Berechne Property-Überlappung
        overlaps = []
        for pattern in patterns:
            pattern_props = set(pattern.properties.keys())
            overlap = len(pattern_props.intersection(all_properties)) / len(all_properties)
            overlaps.append(overlap)
        
        avg_overlap = np.mean(overlaps)
        
        # Prüfe Beziehungs-Ähnlichkeit
        all_relationships = set()
        for pattern in patterns:
            all_relationships.update(pattern.relationships)
        
        rel_overlaps = []
        for pattern in patterns:
            pattern_rels = set(pattern.relationships)
            if all_relationships:
                overlap = len(pattern_rels.intersection(all_relationships)) / len(all_relationships)
                rel_overlaps.append(overlap)
        
        avg_rel_overlap = np.mean(rel_overlaps) if rel_overlaps else 0
        
        # Merge wenn hohe Ähnlichkeit
        return avg_overlap > 0.7 and avg_rel_overlap > 0.5
    
    async def _perform_node_merge(self, patterns: List[NodePattern]) -> Optional[SchemaEvolution]:
        """Führt Knoten-Merge durch.
        
        Args:
            patterns: Zu mergende Knoten-Muster
            
        Returns:
            SchemaEvolution oder None
        """
        if len(patterns) < 2:
            return None
        
        # Wähle Haupt-Label (häufigstes)
        main_pattern = max(patterns, key=lambda p: p.frequency)
        merge_labels = [p.label for p in patterns if p.label != main_pattern.label]
        
        try:
            # Führe Merge in Neo4j durch
            query = """
            MATCH (n)
            WHERE $old_label IN labels(n)
            CALL apoc.create.addLabels(n, [$new_label]) YIELD node
            CALL apoc.create.removeLabels(node, [$old_label]) YIELD node as updated_node
            RETURN count(updated_node) as merged_count
            """
            
            total_merged = 0
            with self.driver.session() as session:
                for old_label in merge_labels:
                    result = session.run(query, {
                        'old_label': old_label,
                        'new_label': main_pattern.label
                    })
                    record = result.single()
                    if record:
                        total_merged += record['merged_count']
            
            if total_merged > 0:
                evolution = SchemaEvolution(
                    timestamp=datetime.now(),
                    evolution_type='node_merge',
                    description=f"Merged {merge_labels} into {main_pattern.label}",
                    affected_nodes=[main_pattern.label] + merge_labels,
                    affected_relationships=[],
                    performance_impact=0.1  # Positive impact
                )
                
                logger.info(f"Merged {total_merged} nodes: {merge_labels} -> {main_pattern.label}")
                return evolution
                
        except Exception as e:
            logger.error(f"Failed to merge nodes: {e}")
        
        return None
    
    async def _optimize_relationships(self) -> List[SchemaEvolution]:
        """Optimiert bestehende Beziehungen.
        
        Returns:
            Liste der Relationship-Optimierungen
        """
        evolutions = []
        
        # Finde redundante Beziehungen
        redundant_rels = await self._find_redundant_relationships()
        
        for rel_info in redundant_rels:
            evolution = await self._remove_redundant_relationship(rel_info)
            if evolution:
                evolutions.append(evolution)
        
        return evolutions
    
    async def _find_redundant_relationships(self) -> List[Dict[str, Any]]:
        """Findet redundante Beziehungen.
        
        Returns:
            Liste redundanter Beziehungen
        """
        query = """
        MATCH (a)-[r1]->(b)-[r2]->(c)
        WHERE type(r1) = type(r2) AND a <> c
        WITH a, c, type(r1) as rel_type, count(*) as path_count
        WHERE path_count > 5
        MATCH (a)-[direct]->(c)
        WHERE type(direct) = rel_type
        RETURN a.id as source_id, c.id as target_id, rel_type, path_count
        """
        
        redundant = []
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                redundant.append({
                    'source_id': record['source_id'],
                    'target_id': record['target_id'],
                    'rel_type': record['rel_type'],
                    'path_count': record['path_count']
                })
        
        return redundant
    
    async def _remove_redundant_relationship(self, rel_info: Dict[str, Any]) -> Optional[SchemaEvolution]:
        """Entfernt redundante Beziehung.
        
        Args:
            rel_info: Information über redundante Beziehung
            
        Returns:
            SchemaEvolution oder None
        """
        try:
            query = """
            MATCH (a {id: $source_id})-[r:$rel_type]->(c {id: $target_id})
            DELETE r
            RETURN count(r) as deleted_count
            """
            
            with self.driver.session() as session:
                result = session.run(query, rel_info)
                record = result.single()
                
                if record and record['deleted_count'] > 0:
                    evolution = SchemaEvolution(
                        timestamp=datetime.now(),
                        evolution_type='relationship_optimization',
                        description=f"Removed redundant {rel_info['rel_type']} relationship",
                        affected_nodes=[rel_info['source_id'], rel_info['target_id']],
                        affected_relationships=[rel_info['rel_type']],
                        performance_impact=0.05
                    )
                    
                    logger.info(f"Removed redundant relationship: {rel_info}")
                    return evolution
                    
        except Exception as e:
            logger.error(f"Failed to remove redundant relationship: {e}")
        
        return None
    
    async def _create_inferred_relationships(self) -> List[SchemaEvolution]:
        """Erstellt neue Beziehungen basierend auf Mustern.
        
        Returns:
            Liste neuer Beziehungen
        """
        evolutions = []
        
        # Finde potenzielle neue Beziehungen
        potential_rels = await self._find_potential_relationships()
        
        for rel_info in potential_rels:
            evolution = await self._create_inferred_relationship(rel_info)
            if evolution:
                evolutions.append(evolution)
        
        return evolutions
    
    async def _find_potential_relationships(self) -> List[Dict[str, Any]]:
        """Findet potenzielle neue Beziehungen.
        
        Returns:
            Liste potenzieller Beziehungen
        """
        query = """
        MATCH (a)-[r1]->(intermediate)-[r2]->(b)
        WHERE NOT (a)-[:RELATED_TO]->(b) AND a <> b
        WITH a, b, type(r1) + '_' + type(r2) as inferred_type, count(*) as strength
        WHERE strength > 3
        RETURN a.id as source_id, b.id as target_id, inferred_type, strength
        ORDER BY strength DESC
        LIMIT 10
        """
        
        potential = []
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                potential.append({
                    'source_id': record['source_id'],
                    'target_id': record['target_id'],
                    'rel_type': 'INFERRED_' + record['inferred_type'],
                    'strength': record['strength']
                })
        
        return potential
    
    async def _create_inferred_relationship(self, rel_info: Dict[str, Any]) -> Optional[SchemaEvolution]:
        """Erstellt inferierte Beziehung.
        
        Args:
            rel_info: Information über neue Beziehung
            
        Returns:
            SchemaEvolution oder None
        """
        try:
            query = """
            MATCH (a {id: $source_id}), (b {id: $target_id})
            CREATE (a)-[r:INFERRED_RELATIONSHIP {
                type: $rel_type,
                strength: $strength,
                created_at: datetime()
            }]->(b)
            RETURN r
            """
            
            with self.driver.session() as session:
                result = session.run(query, rel_info)
                record = result.single()
                
                if record:
                    evolution = SchemaEvolution(
                        timestamp=datetime.now(),
                        evolution_type='relationship_creation',
                        description=f"Created inferred relationship: {rel_info['rel_type']}",
                        affected_nodes=[rel_info['source_id'], rel_info['target_id']],
                        affected_relationships=[rel_info['rel_type']],
                        performance_impact=0.02
                    )
                    
                    logger.info(f"Created inferred relationship: {rel_info}")
                    return evolution
                    
        except Exception as e:
            logger.error(f"Failed to create inferred relationship: {e}")
        
        return None
    
    async def _optimize_properties(self) -> List[SchemaEvolution]:
        """Optimiert Knoten- und Beziehungs-Properties.
        
        Returns:
            Liste der Property-Optimierungen
        """
        evolutions = []
        
        # Finde ungenutzte Properties
        unused_props = await self._find_unused_properties()
        
        for prop_info in unused_props:
            evolution = await self._remove_unused_property(prop_info)
            if evolution:
                evolutions.append(evolution)
        
        return evolutions
    
    async def _find_unused_properties(self) -> List[Dict[str, Any]]:
        """Findet ungenutzte Properties.
        
        Returns:
            Liste ungenutzter Properties
        """
        query = """
        MATCH (n)
        WITH labels(n)[0] as label, keys(n) as props
        UNWIND props as prop
        WITH label, prop, count(*) as usage_count
        WHERE usage_count < 5
        RETURN label, prop, usage_count
        """
        
        unused = []
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                unused.append({
                    'label': record['label'],
                    'property': record['prop'],
                    'usage_count': record['usage_count']
                })
        
        return unused
    
    async def _remove_unused_property(self, prop_info: Dict[str, Any]) -> Optional[SchemaEvolution]:
        """Entfernt ungenutzte Property.
        
        Args:
            prop_info: Information über ungenutzte Property
            
        Returns:
            SchemaEvolution oder None
        """
        # Nur entfernen wenn sehr wenig genutzt
        if prop_info['usage_count'] > 2:
            return None
        
        try:
            query = """
            MATCH (n:$label)
            REMOVE n.$property
            RETURN count(n) as updated_count
            """
            
            with self.driver.session() as session:
                result = session.run(query.replace('$label', prop_info['label']).replace('$property', prop_info['property']))
                record = result.single()
                
                if record and record['updated_count'] > 0:
                    evolution = SchemaEvolution(
                        timestamp=datetime.now(),
                        evolution_type='property_optimization',
                        description=f"Removed unused property {prop_info['property']} from {prop_info['label']}",
                        affected_nodes=[prop_info['label']],
                        affected_relationships=[],
                        performance_impact=0.01
                    )
                    
                    logger.info(f"Removed unused property: {prop_info}")
                    return evolution
                    
        except Exception as e:
            logger.error(f"Failed to remove unused property: {e}")
        
        return None
    
    async def _store_evolution(self, evolution: SchemaEvolution):
        """Speichert Evolution-Event."""
        query = """
        CREATE (e:SchemaEvolution {
            timestamp: $timestamp,
            evolution_type: $evolution_type,
            description: $description,
            affected_nodes: $affected_nodes,
            affected_relationships: $affected_relationships,
            performance_impact: $performance_impact
        })
        """
        
        with self.driver.session() as session:
            session.run(query, {
                'timestamp': evolution.timestamp.isoformat(),
                'evolution_type': evolution.evolution_type,
                'description': evolution.description,
                'affected_nodes': json.dumps(evolution.affected_nodes),
                'affected_relationships': json.dumps(evolution.affected_relationships),
                'performance_impact': evolution.performance_impact
            })
    
    async def get_evolution_summary(self) -> Dict[str, Any]:
        """Gibt Evolution-Zusammenfassung zurück.
        
        Returns:
            Dictionary mit Evolution-Statistiken
        """
        evolution_types = Counter([e.evolution_type for e in self.evolution_history])
        
        recent_evolutions = [e for e in self.evolution_history 
                           if e.timestamp > datetime.now() - timedelta(days=7)]
        
        total_impact = sum([e.performance_impact for e in self.evolution_history])
        
        summary = {
            'total_evolutions': len(self.evolution_history),
            'evolution_types': dict(evolution_types),
            'recent_evolutions': len(recent_evolutions),
            'total_performance_impact': total_impact,
            'node_patterns': len(self.node_patterns),
            'relationship_patterns': len(self.relationship_patterns),
            'clustered_nodes': len([p for p in self.node_patterns if p.cluster_id is not None]),
            'last_evolution': self.evolution_history[0].timestamp.isoformat() if self.evolution_history else None
        }
        
        return summary
    
    async def start_continuous_evolution(self, interval_hours: int = 24):
        """Startet kontinuierliche Schema-Evolution.
        
        Args:
            interval_hours: Intervall in Stunden zwischen Evolution-Läufen
        """
        logger.info(f"Starting continuous schema evolution with {interval_hours} hour intervals")
        
        while True:
            try:
                # Führe Schema-Evolution durch
                evolutions = await self.evolve_schema()
                
                if evolutions:
                    logger.info(f"Completed evolution cycle with {len(evolutions)} changes")
                else:
                    logger.info("No schema changes needed in this cycle")
                
                # Re-analysiere Schema nach Änderungen
                if evolutions:
                    await self._analyze_current_schema()
                
                # Warte bis zum nächsten Intervall
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Error in continuous evolution: {e}")
                await asyncio.sleep(3600)  # 1 Stunde Pause bei Fehlern

# Beispiel-Nutzung
if __name__ == "__main__":
    async def main():
        engine = GraphEvolutionEngine()
        await engine.initialize()
        
        # Führe Evolution durch
        evolutions = await engine.evolve_schema()
        
        # Zeige Zusammenfassung
        summary = await engine.get_evolution_summary()
        print(json.dumps(summary, indent=2, default=str))
        
        await engine.close()
    
    asyncio.run(main())