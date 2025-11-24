# Transcription CRNN

Below are presented the steps needed to:

<ol>
  <li>Add and pre-process new images in the database</li>
  <li>Load the trained model</li>
  <li>Perform new predictions</li>
</ol>

## Steps to Pre-Process new images:

<ol>
  <li>Add the desired full pages in PNG format inside a folder, with the name convention "textoXX_paginaY"</li>
  <li>Inside the pre_processing.ipynb file:</li>
    <ol>
      <li>Check the horizontal projection of the first image from your folder with the cell 10 (replace the folder path with the one being used)</li>
      <li>Run the cell 11, passing as parameters: the path to the folder with the original images, the destination folder, and the threshold (minimum pixel density to consider a new line)</li>
      <li>In the window that pops-up, manually rotate the image and select the desired angle</li>
      <li>Then, go over the cropped lines and confirm if the resulting image is adequate</li>
      <li>Check the results in the output and compare the original images with their result after applying the deslanting algorithm</li>
    </ol>
  <li>The processed images will be sent to the destination folder and can then be used to train the CRNN</li>
</ol>

## Steps to Load and test the trained CRNN:

<ol>
  <li>Inside the test_model_increased.ipynb file:</li>
    <ol>
      <li>In the cell 7, load the trained model using the load_model() method</li>
    </ol>
</ol>
