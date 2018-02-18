from neo4j.v1 import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j"))


session = driver.session()


result = session.run("MATCH (a)-[r]->(b) WITH collect({source: id(a),target: id(b),caption: type(r)}) AS edges RETURN edges")

for record in result:
    print(record["edges"])