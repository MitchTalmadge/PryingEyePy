import pickle

import glob
import os
import base64


class Face:
    """
    Face Data Model
    id: this corresponds directly to the file name
    name: name associated with this Face
    data: the result of face_recognition.face_encodings
    """
    id: str
    name: str
    data: list

    def __init__(self, name, data):
        self.id = base64.b64encode(name.encode('utf-8')).decode('utf-8')
        self.name = name
        self.data = data


class FaceRepository:
    src_folder = "models/"

    def __init__(self):
        if not os.path.exists(self.src_folder):
            os.makedirs(self.src_folder)

    def _get_model_dir(self):
        """
        :param id: str
        :rtype: str
        :returns a model path given the ID
        """
        return os.path.join(os.getcwd(), self.src_folder)

    def _get_model_path(self, id: str):
        """
        :param id: str
        :rtype: str
        :returns a model path given the ID
        """
        return os.path.join(self._get_model_dir(), f"{id}.dat")

    def load(self, id: str):
        """
        Load a Face model given it's ID
        :param id: str - to load
        :return: Face
        """
        with open(self._get_model_path(id), 'rb') as file:
            data = pickle.load(file)

        return Face(base64.b64decode(id.encode('utf-8')).decode('utf-8'), data)

    def load_all(self):
        """
        Loads all Face models in the src_folder
        :return: list - list of Faces saved
        """
        faces = []

        files = []
        for file in glob.glob(self._get_model_dir() + "/*.dat"):
            files.append(file)

        for file in files:
            face_id = os.path.basename(file).replace(".dat", "")
            faces.append(self.load(face_id))

        return faces

    def save(self, face: Face):
        """
        Saves a Face model
        :param face: Face - to save
        """
        with open(self._get_model_path(face.id), 'wb') as file:
            pickle.dump(face.data, file)

    def delete(self, face: Face):
        """
        Deletes a Face model
        :param face: Face - to delete
        """
        os.remove(self._get_model_path(face.id))
