import json
from pathlib import Path
from types import MappingProxyType

def _parse_edge_key(edge_key):
	start, end = edge_key.split(",", 1)
	return int(start), int(end)

def load():
	data_dir = Path(__file__).resolve().parent

	with (data_dir / "Coord.json").open("r", encoding="utf-8") as f:
		raw_coord = json.load(f)
	with (data_dir / "Cost.json").open("r", encoding="utf-8") as f:
		raw_cost = json.load(f)
	with (data_dir / "Dist.json").open("r", encoding="utf-8") as f:
		raw_dist = json.load(f)
	with (data_dir / "G.json").open("r", encoding="utf-8") as f:
		raw_g = json.load(f)

	coord = {int(node): tuple(value) for node, value in raw_coord.items()}
	cost = {_parse_edge_key(edge): weight for edge, weight in raw_cost.items()}
	dist = {_parse_edge_key(edge): length for edge, length in raw_dist.items()}
	g = {int(node): [int(neighbor) for neighbor in neighbors] for node, neighbors in raw_g.items()}

	return (
		MappingProxyType(coord),
		MappingProxyType(cost),
		MappingProxyType(dist),
		MappingProxyType(g),
	)
