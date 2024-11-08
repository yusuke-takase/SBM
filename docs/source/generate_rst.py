import sbm
import inspect
import os

def escape_underscores(name):
    return name.replace('_', '\\_')

def is_special_method(name):
    return name.startswith('__') and name.endswith('__')

def get_classes_and_methods(module):
    classes = inspect.getmembers(module, inspect.isclass)
    for class_name, class_obj in classes:
        escaped_class_name = escape_underscores(class_name)
        class_filename = f"sbm.{class_name}.rst"
        with open(class_filename, 'w') as class_file:
            class_file.write(f"sbm.{escaped_class_name}\n")
            class_file.write("=" * len(f"sbm.{escaped_class_name}") + "\n\n")
            class_file.write(f".. currentmodule:: sbm\n\n")
            class_file.write(f".. autoclass:: {class_name}\n\n")
            class_file.write(f"   .. automethod:: __init__\n\n")
            class_file.write(f"   .. rubric:: Methods\n\n")
            class_file.write(f"   .. autosummary::\n\n")

            methods = inspect.getmembers(class_obj, inspect.isfunction)
            for method_name, method_obj in methods:
                if is_special_method(method_name):
                    continue
                escaped_method_name = escape_underscores(method_name)
                class_file.write(f"      ~{class_name}.{method_name}\n")
                method_filename = f"sbm.{class_name}.{method_name}.rst"
                with open(method_filename, 'w') as method_file:
                    method_file.write(f"sbm.{escaped_class_name}.{escaped_method_name}\n")
                    method_file.write("=" * len(f"sbm.{escaped_class_name}.{escaped_method_name}") + "\n\n")
                    method_file.write(f".. currentmodule:: sbm\n\n")
                    method_file.write(f".. automethod:: {class_name}.{method_name}\n")

def get_functions(module):
    functions = inspect.getmembers(module, inspect.isfunction)
    for function_name, function_obj in functions:
        if is_special_method(function_name):
            continue
        escaped_function_name = escape_underscores(function_name)
        function_filename = f"sbm.{function_name}.rst"
        with open(function_filename, 'w') as function_file:
            function_file.write(f"sbm.{escaped_function_name}\n")
            function_file.write("=" * len(f"sbm.{escaped_function_name}") + "\n\n")
            function_file.write(f".. currentmodule:: sbm\n\n")
            function_file.write(f".. autofunction:: {function_name}\n")

def generate_reference_rst(module, output_file):
    with open(output_file, 'w') as f:
        f.write("API Reference\n")
        f.write("=============\n\n")

        # Main functions
        f.write("Main functions\n")
        f.write("--------------\n\n")
        f.write(".. toctree::\n")
        f.write(".. autosummary::\n")
        f.write("   :toctree: generated/\n\n")

        functions = inspect.getmembers(module, inspect.isfunction)
        for function_name, function_obj in functions:
            if not is_special_method(function_name):
                f.write(f"   sbm.{function_name}\n")
        f.write("\n")

        # Classes
        f.write("Classes\n")
        f.write("-------\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 2\n\n")
        f.write(".. autosummary::\n")
        f.write("   :toctree: generated/\n\n")

        classes = inspect.getmembers(module, inspect.isclass)
        for class_name, class_obj in classes:
            f.write(f"   sbm.{class_name}\n")
        f.write("\n")

        # Methods
        f.write("Methods\n")
        f.write("-------\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 2\n\n")
        f.write(".. autosummary::\n")
        f.write("   :toctree: generated/\n")
        f.write("   :recursive:\n\n")

        for class_name, class_obj in classes:
            methods = inspect.getmembers(class_obj, inspect.isfunction)
            for method_name, method_obj in methods:
                if not is_special_method(method_name):
                    f.write(f"   sbm.{class_name}.{method_name}\n")
        f.write("\n")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = "generated"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = "reference.rst"
    generate_reference_rst(sbm, output_file)
    print(f"{output_file} has been generated successfully.")

    os.chdir(output_dir)

    print("Generating documentation files for sbm module...")

    get_classes_and_methods(sbm)
    get_functions(sbm)



    print("Documentation files generated successfully.")
