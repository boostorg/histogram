# Contributing to Boost.Histogram

## Star the project

If you like Boost.Histogram, please star the project on Github! We want Boost.Histogram to be the best histogram library out there. If you give it a star, it becomes more visible and will gain more users. More users mean more user feedback to make the library even better.

## Reporting Issues

We value your feedback about issues you encounter. The more information you provide the easier it is for developers to resolve the problem.

Issues should be reported to the [issue tracker](
https://github.com/boostorg/histogram/issues?state=open).

Issues can also be used to submit feature requests.

And don't be shy: if you are friendly, we are friendly! And we care, issues are usually answered within a working day.

## Submitting Pull Requests

Base your changes on `develop`. The `master` branch is only used for releases.

Please rebase your changes on the current develop branch before submitting (which may have diverged from your fork in the meantime). This keeps the git history cleaner and easier to understand.

## Coding Style

* Use clang-format -style=file, which should pick up the .clang-format file of the project

## Running Tests

To build the tests you can use `cmake` or `b2`. If you use `cmake`, the tests can be run by executing the `ctest` command from the build directory. To run the tests with `b2`, just go the test folder and execute `b2`.

Please report any tests failures to the issue tracker along with the test
output and information on your system and compute device.

## Support

Feel free to send an email to hans.dembinski@gmail.com with any problems or questions.
