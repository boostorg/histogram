git co gh-pages
git merge develop -m automerge
rm -rf html
b2
git ci html -m update
git co develop
