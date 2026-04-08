import unittest

from dilu.driver_agent.vectorStore import DrivingMemory


class _DummyScenario:
    def describe(self, frame_id):
        return f"frame {frame_id}"


class _DummyCollection:
    def __init__(self, count_value):
        self._count_value = count_value

    def count(self):
        return self._count_value


class _DummyScenarioMemory:
    def __init__(self, count_value):
        self._collection = _DummyCollection(count_value)
        self.similarity_called = False

    def similarity_search_with_score(self, query, k):
        self.similarity_called = True
        raise AssertionError("similarity_search_with_score should not be called for an empty memory store")


class VectorStoreTests(unittest.TestCase):
    def test_retrive_memory_short_circuits_when_collection_is_empty(self):
        memory = object.__new__(DrivingMemory)
        memory.encode_type = "sce_language"
        memory.scenario_memory = _DummyScenarioMemory(count_value=0)

        result = memory.retriveMemory(_DummyScenario(), frame_id=0, top_k=1)

        self.assertEqual(result, [])
        self.assertFalse(memory.scenario_memory.similarity_called)


if __name__ == "__main__":
    unittest.main()
