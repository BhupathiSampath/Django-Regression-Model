
from django import forms
 
# creating a form
class InputForm(forms.Form):
 
    name = forms.CharField(required=False)
    email = forms.EmailField()
    message = forms.CharField(max_length=1000)