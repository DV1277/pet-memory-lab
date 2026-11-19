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

    def test_exact_lookup_keeps_subjects_distinct_after_update(self):
        owner = self.memory.add_fact(
            "I like jazz.", subject="Carlos", detail="favorite music"
        )
        pet = self.memory.add_fact(
            "Maple likes rock.", subject="Maple", detail="favorite music"
        )

        self.assertEqual(
            self.memory.lookup_fact(" CARLOS ", "favorite  MUSIC"), owner
        )
        self.assertEqual(self.memory.lookup_fact("Maple", "favorite music"), pet)

        updated = self.memory.update_fact(owner["id"], "I like blues.")

        self.assertEqual(updated["id"], owner["id"])
        self.assertEqual(
            self.memory.lookup_fact("Carlos", "favorite music"),
            {"id": owner["id"], "fact": "I like blues."},
        )
        self.assertEqual(self.memory.lookup_fact("Maple", "favorite music"), pet)

    def test_exact_lookup_returns_none_for_missing_detail(self):
        self.memory.add_fact(
            "I like jazz.", subject="Carlos", detail="favorite music"
        )

        self.assertIsNone(self.memory.lookup_fact("Carlos", "passport number"))

    def test_exact_lookup_rejects_ambiguous_shared_detail(self):
        shared = self.memory.add_fact(
            "Maple and Carlos hike.",
            subject="Maple and Carlos",
            detail="shared activity",
        )
        self.assertEqual(
            self.memory.lookup_fact("Maple and Carlos", "shared activity"), shared
        )

        self.memory.add_fact(
            "Maple and Carlos swim.",
            subject="Maple and Carlos",
            detail="shared activity",
        )
        with self.assertRaisesRegex(ValueError, "Multiple facts"):
            self.memory.lookup_fact("Maple and Carlos", "shared activity")


if __name__ == "__main__":
    unittest.main(verbosity=2)