# The Georgia project on https://github.com/KatherineMossDeveloper/The-Georgia-Project/tree/main
# GA_weaviatedatabase.py
#
# class WeaviateDatabase
#     def weaviate_connect(self)  weaviate-client version 3.24.2.
#     def weaviate_available()
#     def weaviate_delete_and_create_schema()
#     def weaviate_truncate(self)
#     def weaviate_add_record(self, filename, image_vector)
#     def weaviate_find_neighbors(self, image_vector, limit)
#     def weaviate_row_count(self)
#     def weaviate_fetch_record()
#
# This code will connected with and maintain a weaviate database, if one is up and running.
#
# To do.
# (nothing)
# #############################################################################################
import weaviate
import uuid


class WeaviateDatabase:

    def __init__(self,
                 class_name="CrystalImage",
                 class_description="A crystal image feature vector with metadata",
                 class_vectorizer="none"):
        self.weaviate_connected = False
        self.weaviate_class_name = class_name
        self.weaviate_class_desc = class_description
        self.weaviate_class_vectorizer = class_vectorizer
        self.client_connection = None

    def weaviate_connect(self):

        try:
            self.client_connection = weaviate.Client("http://localhost:8080")

            if self.client_connection.is_ready():
                print("Weaviate_connect:  weaviate is ready.")
                self.weaviate_connected = True
            else:
                print("Weaviate_connect:  weaviate is not ready.")
                self.weaviate_connected = False
            return True

        except Exception as e:
            print(f"Error thrown in GA_dataprocessing.weaviate_connect:  {e}")
            return False

    def weaviate_available(self):

        try:

            # see if the server is ready
            if not self.weaviate_connect():
                print("Weaviate is not ready.")
                return False
            else:
                print(f"weaviate-client_connection version: {weaviate.__version__}")

            # print schema, if available and needed.
            # schema = self.client_connection.schema.get()
            # print(json.dumps(schema, indent=2))
            return True

        except Exception as e:
            print(f"GA_dataprocessing.weaviate_available:  could not connect to Weaviate: {e}")
            return False

    def weaviate_delete_and_create_schema(self):

        try:
            print("Inside weaviate_create_schema.")

            # step 0.  connect to local Weaviate, if it is available.
            if self.weaviate_connect():

                # step 1.  delete schema.
                class_name = self.weaviate_class_name
                if self.client_connection.schema.exists(class_name):
                    self.client_connection.schema.delete_class(class_name)

                # step 2.  create new scheme.
                schema = {
                    "class": class_name,
                    "description": self.weaviate_class_desc,
                    "vectorizer": self.weaviate_class_vectorizer,
                    "properties": [
                        {"name": "image_id", "dataType": ["string"], "description": "The image file name or ID"},
                        {"name": "class_label", "dataType": ["string"], "description": "The image label."},
                        {"name": "confidence", "dataType": ["number"], "description": "Predicted confidence %"}
                    ]
                }

                self.client_connection.schema.create_class(schema)
                print(f"Created class {class_name}")

                return True
            return False

        except Exception as e:
            print(f"Error thrown in GA_dataprocessing.weaviate_create_schema:  {e}")
            return False

    def weaviate_truncate(self):

        try:
            # see if the server is ready
            if self.weaviate_connect:
                self.client_connection.batch.delete_objects(
                    class_name="CrystalImage",
                    where={
                        "path": ["image_id"],
                        "operator": "Like",
                        "valueText": "*"
                    }
                )
            return True

        except Exception as e:
            print(f"Error thrown in GA_dataprocessing.weaviate_truncate:  {e}")
            return False

    def weaviate_add_record(self, filename, class_label, confidence_factor, image_vector):

        try:
            if not self.weaviate_connect():
                print("No connection to the database.")

            else:
                data_objects = [
                    {
                        "image_id": filename,                    # string
                        "class_label": class_label,              # string
                        "confidence": float(confidence_factor),  # float
                        "_id": str(uuid.uuid4())
                    }
                ]

                for obj in data_objects:
                    self.client_connection.data_object.create(
                        data_object={
                            "image_id": obj["image_id"],
                            "class_label": obj["class_label"],
                            "confidence": obj["confidence"]
                        },
                        class_name=self.weaviate_class_name,
                        uuid=obj["_id"],
                        vector=image_vector
                    )
                    print(f"Added record for {obj['image_id']}")

                return True

            return False

        except Exception as e:
            print(f"Error thrown in GA_dataprocessing.weaviate_add_record:  {e}")
            return False

    def weaviate_find_neighbors(self, image_vector, limit):

        try:
            # see if the server is ready
            if self.weaviate_connect():
                result = self.client_connection.query.get(
                    "CrystalImage",
                    ["image_id", "_additional { distance }"]
                ).with_near_vector({
                    "vector": image_vector
                }).with_limit(limit).do()

                print("\nThese neighbors were found:")
                for hit in result['data']['Get'][self.weaviate_class_name]:
                    print(hit)

                return result['data']['Get'][self.weaviate_class_name]

        except Exception as e:
            print(f"Error thrown in GA_dataprocessing.weaviate_find_neighbors:  {e}")
            return False

    def weaviate_row_count(self):

        try:
            if not self.weaviate_connect():
                print("No connection to the database.")

            else:
                result = self.client_connection.query.aggregate("CrystalImage").with_meta_count().do()
                count = result['data']['Aggregate']['CrystalImage'][0]['meta']['count']
                print(f"Total objects in CrystalImage: {count}")

        except Exception as e:
            print(f"Error thrown in GA_dataprocessing.weaviate_row_count:  {e}")
            return False

    @staticmethod
    def weaviate_fetch_record():

        # This will fetch the first vector, if there is one, in the db.  It is for testing.
        try:
            client = weaviate.Client(
                url="http://localhost:8080",
            )
            result = client.query.get(
                        "CrystalImage",
                        ["image_id", "_additional { vector }"]
                    ).with_limit(1).do()

            # Extract vector
            first_object = result["data"]["Get"]["CrystalImage"][0]
            vector = first_object["_additional"]["vector"]
            print(f"Image ID: {first_object['image_id']}")
            print(f"Vector length: {len(vector)}")
            return vector

        except Exception as e:
            print(f"Error thrown in GA_dataprocessing.weaviate_fetch_record:  {e}")
            return False

    @staticmethod
    def weaviate_select_records(limit):

        # This will fetch records and print them.
        try:
            client = weaviate.Client(
                url="http://localhost:8080",
            )

            result = (
                client.query
                .get(
                    "CrystalImage",
                    [
                        "image_id",
                        "class_label",
                        "confidence",
                    ]
                )
                .with_additional(["id", "vector"])  # <--- this is the key
                .with_limit(limit)
                .do()
            )

            # print results, if needed, for debugging.
            # for item in result['data']['Get']['CrystalImage']:
            #    print(item)

            return result

        except Exception as e:
            print(f"Error thrown in GA_dataprocessing.weaviate_select_records:  {e}")
            return False
