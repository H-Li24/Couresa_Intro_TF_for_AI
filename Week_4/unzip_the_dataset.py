import zipfile

local_zip = './validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip,'r')
zip_ref.extractall('./validation-horse-or-human')
zip_ref.close()