import apache_beam as beam
from google.cloud import storage
import urllib.request


class LoadImagesToGcs(beam.DoFn):
    def process(
                self,
                image_url,
                bucket_name,
                bucket_file_path
               ):
        """
        Uploads an image from a URL source to GCS bucket/blob
        Args:
            image_url: string URL of the image e.g. https://picsum.photos/200/200
            bucket_name: name of GCS bucket where images are to uploaded
            bucket_file_path: relative path of folder (inside GCS bucket) where images are to uploaded
        """

        # Get appropriate/unique name for each image file to be stored
        split_image_url = image_url.split(':')
        image_name = split_image_url[-1]
        image_name = image_name.replace("-", "_")

        # separation of bucket name from blob name
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(bucket_file_path + "/" + image_name + ".jpg")

        # try to read the image URL
        try:
            print(f'Getting image from: {image_url}')
            with urllib.request.urlopen(image_url) as response:

                # check if URL contains an image
                info = response.info()
                if (info.get_content_type().startswith("image")):  # e.g. 'image/jpeg'
                    blob.upload_from_string(response.read(),
                                            content_type=info.get_content_type())
                    print("Uploaded image from: " + image_url)
                else:
                    print("Could not upload image. No image data type in URL")

        except urllib.error.HTTPError as exception:
            print(f'Could not upload image from {image_url}. Error message: {exception}')

        return


PROJECT = "kubeflow-1-0-2"  # <-- Change this
bucket_name = "fire_detection_anurag"  # <-- Change this
bucket_file_path = "no_fire/images"  # <-- Change this
full_path = bucket_name + "/" + bucket_file_path
file_csv = "gs://fire_detection_anurag/no_fire/No_fire_combined.csv"  # <-- Change this
dependencies = "requirements.txt"
max_num_workers = 5


def run():
    argv = [
            '--project={0}'.format(PROJECT),
            '--job_name=upload-to-gcs',
            '--save_main_session',
            '--max_num_workers={0}'.format(max_num_workers),
            '--requirements_file={0}'.format(dependencies),
            '--region=us-central1',
            '--runner=DataflowRunner'
           ]
    # Note: Better to explicitly use 'staging_location' and 'temp_location' too.
    # Reason: Otherwise it can create tmp/staging folders which persist in our GCS buckets
    # and might break the downstream parts of the code.

    p = beam.Pipeline(argv=argv)

    (
     p
     | beam.io.ReadFromText(file_csv)
     | beam.ParDo(
                  LoadImagesToGcs(),
                  # side inputs
                  bucket_name,
                  bucket_file_path
                 )
    )

    p.run()


if __name__ == '__main__':
    run()
