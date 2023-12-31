# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound


@blueprint.route('/index', methods=('GET', 'POST'))
#@login_required
def index():
    messages = []
    if request.method == 'POST':
        if request.form['usercontent']:
            myinput = request.form['usercontent']


            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            model_id = "codellama/CodeLlama-7b-Instruct-hf"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to("cuda")


            system = "Provide answers in Python. Display only code. Display only code in ```. Respond only with code. Do not include comments. Do not give details about the code. Make sure that the answer contains Python's print statement to print out the results. "
            user = myinput

            prompt = f"<s>[INST] <<SYS>>\\n{system}\\n<</SYS>>\\n\\n{user}[/INST]"
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

            output = model.generate(
                inputs["input_ids"],
                max_new_tokens=200,
                do_sample=True,
                top_p=0.9,
                temperature=0.1,
            )
            output = output[0].to("cpu")
            sout = tokenizer.decode(output).split("[/INST]",1)[1][:-4].strip()
            sout = find_between( sout, "```", "```" )


            import sys
            from io import StringIO
            import contextlib

            @contextlib.contextmanager
            def stdoutIO(stdout=None):
                old = sys.stdout
                if stdout is None:
                    stdout = StringIO()
                sys.stdout = stdout
                yield stdout
                sys.stdout = old

            with stdoutIO() as s:
                try:
                    exec(sout)
                except:
                    pass 

            if s.getvalue():
                sout = s.getvalue()
            messages.append({'output': sout})
    return render_template('home/index.html', segment='index', messages=messages)


def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end].strip()
    except ValueError:
        return ""

@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
