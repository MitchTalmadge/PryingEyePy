import unittest
import os

import numpy as np

from src.face import Face, FaceRepository


class FaceTest(unittest.TestCase):

    def setUp(self):
        self.my_repo = FaceRepository()

        self.id = "R3V5YnJ1c2ggVGhyZWVwd29vZA=="
        self.name = "Guybrush Threepwood"
        self.data = np.array([1, 2, 3])
        self.expected_file_path = f"models/{self.id}.dat"

    def test_save(self):
        my_face = Face(self.name, self.data)

        self.my_repo.save(my_face)

        self.assertTrue(os.path.isfile(self.expected_file_path), "File created")

    def test_load(self):
        self.test_save()

        my_face = self.my_repo.load(self.id)

        self.assertEqual(my_face.name, self.name)
        np.testing.assert_array_equal(my_face.data, np.array([1, 2, 3]))

    def test_load_all(self):
        self.test_save()

        my_faces = self.my_repo.load_all()

        self.assertEqual(len(my_faces), 1)

    def test_delete(self):
        self.test_save()
        my_face = Face(self.name, self.data)

        self.my_repo.delete(my_face)

        self.assertFalse(os.path.isfile(self.expected_file_path), "File deleted")


if __name__ == '__main__':
    unittest.main()
