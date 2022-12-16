import argparse


class CLargs:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Optional arguments for run", prog="Run.py")

        parser.add_argument("-H", "--Help", help="Example: Help argument", required=False, default="")
        parser.add_argument(
            "-PopulationSize", help="PopulationSize: Positive integer", required=False, type=int, default="100"
        )
        parser.add_argument("-Iterations", help="Iterations: Positive integer", required=False, type=int, default="100")
        parser.add_argument(
            "-MutationRate", help="MutationRate: Float in (0,1)", required=False, type=float, default="0.01"
        )
        parser.add_argument(
            "-SelectionProportion",
            help="SelectionProportion: Float in (0,1)",
            type=float,
            required=False,
            default="0.01",
        )

        argument = parser.parse_args()
        status = False

        # Set defaults
        self.set_defaults()

        self.report_passed_args(argument)

    def set_defaults(self):
        self.p = 100
        self.i = 100
        self.m = 0.01
        self.s = 0.1

    def print_args(self):
        print("PopulationSize: {}".format(self.p))
        print("Iterations:{}".format(self.i))
        print("MutationRate:{}".format(self.m))
        print("SelectionProportion:{}".format(self.s))

    def return_args(self):
        return self.p, self.i, self.m, self.s

    def report_passed_args(self, argument):
        args_dict = vars(argument)
        arg_names = list(args_dict.keys())
        arg_vals = list(args_dict.values())
        defaults = [self.p, self.i, self.m, self.s]

        if argument.Help:
            print("You have used '-H' or '--Help' with argument: {0}".format(argument.Help))
            status = True

        if argument.PopulationSize:
            self.p = int(argument.PopulationSize)
            print("PopulationSize: {0}".format(self.p))
            status = True
        else:
            print("PopulationSize: {0} (Default)".format(self.p))

        if argument.Iterations:
            self.i = int(argument.Iterations)
            print("Iterations: {0}".format(self.i))
            status = True
        else:
            print("Iterations: {0} (Default)".format(self.i))

        if argument.MutationRate:
            self.m = float(argument.MutationRate)
            print("MutationRate: {0}".format(self.m))
            status = True
        else:
            print("MutationRate: {0} (Default)".format(self.m))

        if argument.SelectionProportion:
            self.s = float(argument.SelectionProportion)
            print("SelectionProportion: {0}".format(self.s))
            status = True
        else:
            print("SelectionProportion: {0} (Default)".format(self.s))

        if not status:
            print("No arguments passed, running with defaults:")
            self.print_args()

