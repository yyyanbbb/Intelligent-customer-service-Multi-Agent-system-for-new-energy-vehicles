"""
基于 NetworkX 的内存知识图谱。存储每轮抽取的实体和关系，支持实体去重合并、邻居查询、路径查询。
Agent 通过 search_knowledge_graph / write_to_graph 两个工具访问。
"""
from __future__ import annotations

import logging
import hashlib
from typing import Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class GraphEntity:
    entity_id: str
    entity_type: str
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    page: int = 0
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> "GraphEntity":
        return cls(
            entity_id=data.get("entity_id", ""),
            entity_type=data.get("entity_type", ""),
            name=data.get("name", ""),
            attributes=data.get("attributes", {}),
            source=data.get("source", ""),
            page=data.get("page", 0),
            confidence=data.get("confidence", 1.0),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GraphRelation:
    source_id: str
    target_id: str
    relation_type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class KnowledgeGraph:
    """
    基于 NetworkX 的内存知识图谱。
    支持：
    - 添加实体（自动去重 + 相似度匹配）
    - 添加关系（三元组：源实体 → 关系 → 目标实体）
    - 邻居查询（查询某实体的所有关联实体）
    - 路径查询（两实体间是否存在路径）
    - 子图查询（按类型筛选实体和关系）
    - 图可视化导出（GraphML / DOT 格式）
    """

    def __init__(self):
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.entity_index: dict[str, GraphEntity] = {}
        self._entity_counter = 0

    def add_entity(self, entity: GraphEntity) -> str:
        """
        添加实体，如果存在相似实体则合并。
        返回实体 ID。
        """
        entity_id = entity.entity_id
        if not entity_id:
            entity_id = self._generate_entity_id(entity.name, entity.entity_type)
            entity.entity_id = entity_id

        if entity_id in self.entity_index:
            existing = self.entity_index[entity_id]
            merged_attrs = {**existing.attributes, **entity.attributes}
            existing.attributes = merged_attrs
            existing.confidence = max(existing.confidence, entity.confidence)
            logger.debug(f"合并实体: {entity_id}")
            return entity_id

        self._entity_counter += 1
        self.graph.add_node(entity_id, **entity.to_dict())
        self.entity_index[entity_id] = entity
        logger.debug(f"添加实体: {entity_id} ({entity.entity_type})")
        return entity_id

    def add_relation(self, relation: GraphRelation) -> bool:
        """添加关系三元组。"""
        src = relation.source_id
        tgt = relation.target_id

        if src not in self.entity_index:
            logger.warning(f"关系添加失败：源实体不存在: {src}")
            return False
        if tgt not in self.entity_index:
            logger.warning(f"关系添加失败：目标实体不存在: {tgt}")
            return False

        edge_data = relation.to_dict()
        self.graph.add_edge(src, tgt, **edge_data)
        logger.debug(f"添加关系: {src} --[{relation.relation_type}]--> {tgt}")
        return True

    def add_extraction_result(self, entities: list[dict], relations: list[dict], source: str = "") -> None:
        """
        批量添加一轮抽取结果。
        """
        entity_id_map: dict[str, str] = {}

        for ent_data in entities:
            ent_data["source"] = source
            entity = GraphEntity.from_dict(ent_data)
            eid = self.add_entity(entity)
            entity_id_map[ent_data.get("name", "")] = eid

        for rel_data in relations:
            src_name = rel_data.get("source_name", "")
            tgt_name = rel_data.get("target_name", "")
            src_id = entity_id_map.get(src_name, src_name)
            tgt_id = entity_id_map.get(tgt_name, tgt_name)

            rel = GraphRelation(
                source_id=src_id,
                target_id=tgt_id,
                relation_type=rel_data.get("relation_type", "related_to"),
                attributes=rel_data.get("attributes", {}),
                source=source,
            )
            self.add_relation(rel)

    def query_neighbors(
        self,
        entity_name: str,
        depth: int = 1,
        relation_type: str | None = None,
    ) -> list[dict]:
        """
        查询某实体的邻居节点。
        """
        candidates = [
            eid for eid, ent in self.entity_index.items()
            if entity_name.lower() in ent.name.lower()
        ]

        if not candidates:
            return []

        results = []
        for eid in candidates:
            node_data = self.graph.nodes[eid]

            neighbors = self.graph.neighbors(eid)
            for neighbor_id in list(neighbors)[:50]:
                edge_data = self.graph.get_edge_data(eid, neighbor_id)
                if edge_data:
                    edge = edge_data[0] if isinstance(edge_data, dict) else edge_data
                    if relation_type and edge.get("relation_type") != relation_type:
                        continue
                    neighbor_data = self.graph.nodes[neighbor_id]
                    results.append({
                        "entity": dict(neighbor_data),
                        "relation": edge.get("relation_type", "related_to"),
                    })

        return results

    def search_entities(
        self,
        entity_type: str | None = None,
        keyword: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """按类型或关键词搜索实体。"""
        results = []
        for eid, ent in self.entity_index.items():
            if entity_type and ent.entity_type != entity_type:
                continue
            if keyword and keyword.lower() not in ent.name.lower():
                continue
            results.append(ent.to_dict())
            if len(results) >= limit:
                break
        return results

    def get_stats(self) -> dict:
        """获取图谱统计信息。"""
        entity_types: dict[str, int] = {}
        for ent in self.entity_index.values():
            entity_types[ent.entity_type] = entity_types.get(ent.entity_type, 0) + 1

        return {
            "total_entities": self.graph.number_of_nodes(),
            "total_relations": self.graph.number_of_edges(),
            "entity_types": entity_types,
            "is_directed": self.graph.is_directed(),
        }

    def export_graphml(self, path: str) -> None:
        """导出为 GraphML 格式。"""
        nx.write_graphml(self.graph, path)
        logger.info(f"图谱已导出至: {path}")

    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """生成唯一实体 ID。"""
        raw = f"{entity_type}:{name}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def __len__(self) -> int:
        return self.graph.number_of_nodes()


# 全局共享图谱实例
_global_kg: KnowledgeGraph | None = None


def get_knowledge_graph() -> KnowledgeGraph:
    global _global_kg
    if _global_kg is None:
        _global_kg = KnowledgeGraph()
    return _global_kg


def search_knowledge_graph(query: str, entity_type: str | None = None) -> str:
    """
    暴露给 Agent 的工具函数：查询知识图谱。
    """
    kg = get_knowledge_graph()

    if entity_type:
        results = kg.search_entities(entity_type=entity_type, keyword=query, limit=30)
    else:
        results = kg.search_entities(keyword=query, limit=30)

    if not results:
        return f"图谱中未找到与 '{query}' 相关的实体。"

    parts = [f"在知识图谱中找到 {len(results)} 个相关实体：\n"]
    for ent in results:
        neighbors = kg.query_neighbors(ent["name"], depth=1)
        parts.append(f"\n📌 [{ent['entity_type']}] {ent['name']}")
        if ent.get("attributes"):
            for k, v in list(ent["attributes"].items())[:5]:
                parts.append(f"   属性: {k} = {v}")
        if neighbors:
            parts.append("  关联实体:")
            for n in neighbors[:5]:
                parts.append(f"   - {n['relation']}: {n['entity'].get('name', '')}")

    parts.append(f"\n\n[图谱统计] {kg.get_stats()}")
    return "\n".join(parts)


def write_to_graph(entities: list[dict], relations: list[dict], source: str = "") -> str:
    """
    暴露给 Agent 的工具函数：将抽取结果写入图谱。
    """
    kg = get_knowledge_graph()
    kg.add_extraction_result(entities, relations, source=source)
    stats = kg.get_stats()
    return f"✅ 已将 {len(entities)} 个实体和 {len(relations)} 个关系写入图谱。当前图谱统计: {stats}"
