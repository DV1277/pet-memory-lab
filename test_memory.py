import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load a copy of memory.py so its database is in a temporary folder.
        cls.temp_dir = tempfile.TemporaryDirectory()
        copied_file = Path(cls.temp_dir.name) / "memory.py"
        shutil.copy2(Path(__file__).with_name("memory.py"), copied_file)

        spec = importlib.util.spec_from_file_location("test_backend", copied_file)
        cls.memory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.memory)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        collection_name = "test_" + self.memory.uuid.uuid4().hex
        self.memory.collection = self.memory.client.create_collection(
            name=collection_name
        )

    def test_empty_database_returns_none(self):
        result = self.memory.search_facts("What music do I like?")
        self.assertIsNone(result)

    def test_dog_question_returns_dog_fact(self):
        self.memory.add_fact("I like listening to jazz music.")
        dog = self.memory.add_fact("My dog likes listening to rock music.")

        result = self.memory.search_facts("What music does my dog like?")

        self.assertEqual(result["id"], dog["id"])

    def test_owner_question_returns_owner_fact(self):
        owner = self.memory.add_fact("I like listening to jazz music.")
        self.memory.add_fact("My dog likes listening to rock music.")

        result = self.memory.search_facts(
            "What type of music does the owner prefer?"
        )

        self.assertEqual(result["id"], owner["id"])
    
    def test_unrelated_question_returns_no_memory(self):
        self.memory.add_fact("My dog likes listening to rock music.")

        result = self.memory.search_facts("What is my passport number?")

        self.assertIsNone(result)

    def test_update_replaces_old_fact(self):
        saved = self.memory.add_fact("My favorite drink is matcha.")

        self.memory.update_fact(saved["id"], "My favorite drink is coffee.")

        facts = self.memory.list_facts()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["id"], saved["id"])
        self.assertEqual(facts[0]["fact"], "My favorite drink is coffee.")
