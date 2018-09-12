# Shennong : Une toolbox pour la reconnaissance non-supervisée de la parole

## Contexte : état des lieux et positionnement avant l’ADT


Dans le domaine du traitement automatique de la parole, les
technologies actuelles nécessitent une grande masse de données
textuelles pour entraîner les modèles acoustiques (des milliers
d’heures de parole transcrites orthographiquement) et les modèles de
langage (textes de milliards de mots). Ce paradigme est coûteux et ne
peut raisonnablement couvrir qu’une faible fraction des langues
parlées dans le monde (environ 7000). Par ailleurs, la moitié au moins
de ces langues ne dispose pas de système orthographique stable et
répandu, rendant impossible l’approche classique.

Ainsi la recherche porte un intérêt croissant aux technologies “zéro
ressource”, qui s’appuient sur l’apprentissage non supervisé de
modèles acoustiques et de modèles de langage à partir d’un signal de
parole brute et non-annotée. Dans cette approche, deux tâches
principales ont été étudiées. La première est l’apprentissage d’une
représentation des unités phonémiques constitutifs d’un langage
(subword modeling), cette représentation devant faciliter la
discrimination des mots tout en étant robuste aux variations intra- et
inter-locuteurs. La seconde tâche est l’apprentissage de mots ou
proto-mots (term discovery), c’est-à-dire l’identification de
fragments de parole récurrents et leur regroupement dans un
dictionnaire. Des études ont démontré que ces deux tâches sont liées :
la connaissance de l’une facilite l’apprentissage de l’autre, et
réciproquement.

L’apprentissage machine en général, et le traitement automatique du
langage en particulier, ont grandement bénéficié de l’existence
d’outils et de bases de données communes et standardisées permettant
l’accès au plus grand nombre et par là favorisant la réplication et la
comparaison des résultats à travers les équipes. Par exemple, en
reconnaissance automatique de la parole, l’existence de datasets
ouverts comme Librivox/Librispeech et de code open source comme HTK et
Kaldi ont favorisé la dissémination et la popularisation de la
recherche au niveau mondial. Le domaine zéro ressource, encore récent,
ne dispose pas de framework ni d’outils dédiés. Chaque équipe
travaille de manière isolée sur ses propres pipelines, outils
d’évaluation et bases de données bien souvent propriétaires. Ainsi les
modèles publiés sont rarement comparés ou répliqués, ni validés sur de
nouveaux langages.

Dans le but de rassembler ces outils hétérogènes et dispersés, une
première démarche a été entreprise par l’équipe CoML au travers de
l’organisation du [Zero Resource Speech
Challenge](http://zerospeech.com) (deux éditions en 2015 et 2017). Une
trentaine de participants a proposé des modèles non-supervisés de
subword modeling et de term discovery, pour lesquels les données et la
méthode d’évaluation étaient imposées, permettant ainsi la comparaison
des résultats. Forts de l’expérience emmagasinée par l’organisation de
ces challenges, qui a déjà permis de fixer des standards d’évaluation,
nous souhaitons à présent passer à l’étape suivante et, dans le cadre
de la présente ADT, développer une toolbox dédiée au zero resource
incluant les meilleurs algorithmes issus de ces deux challenges. Cette
ADT, dont le nom de code est Shennong[1], sera publiée sous licence
libre et disponible sur notre page github :
https://github.com/bootphon. La toolbox se compose de 3 modules
principaux comme illustrée ci-dessous. La planification prévisionnelle
introduit plus de détails.

                                          +--------------------+
                                          |                    |
                                      +-> |  subword modeling  | <---+  evaluation
                                      |   |                    |
                      +------------+  |   +----------+---------+
                      |            |  |              ^
    raw speech  +---> |  features  +--+              |
                      |            |  |              v
                      +------------+  |   +----------+---------+
                                      |   |                    |
                                      +-> |   term discovery   | <---+  evaluation
                                          |                    |
                                          +--------------------+


## Objectifs de l'ADT

Cette ADT a pour but la conception d’une toolbox dédiée à
l’apprentissage non supervisé de la parole pour faciliter la recherche
dans ce domaine. Baptisée Shennong, cette toolbox intégrera les
modèles les plus représentatifs des challenges Zero Resource organisés
par l’équipe, ainsi que des outils d’évaluation des modèles. Conforme
aux standards du logiciel libre, et s’inspirant de ce qui fonctionne
déjà dans le domaine supervisé, Shennong sera ouvert aux contributions
et facilitera la réplication, la comparaison et l’interopérabilité des
modèles. L’ADT se concluera par l’organisation d’une nouvelle
itération du challenge qui exploitera la toolbox et permettra de
l’enrichir de nouveaux modèles.


## Mise en œuvre prévisionnelle de l'ADT

L’ADT s’étale sur 3 ans et se décompose en 6 jalons principaux, chacun
étalé sur environ 6 H/M. Les 3 premiers jalons consistent en
l’implémentation des blocs de traitement, les 3 derniers valideront la
toolbox par une étude comparative des systèmes implémentés et par
l’organisation d’une nouvelle édition du challenge *Zero Resource*.

Le projet suivra les bonnes pratiques de développement dans un
contexte open source. Cela incluera des commits réguliers sur github,
une intégration continue (tests unitaires et tests de réplication) et
une documentation complète en ligne. Nous prévoyons une release de
versions beta du projet au fil de la complétion des jalons, avec une
publication de la version 1.0 de la toolbox au terme de l’ADT.

La toolbox sera implémentée en Python, utilisable en ligne de commande
et comme une bibliothèque Python. Les plateformes visées sont Linux et
MacOS. Elle supportera l’accélération GPU et l’utilisation sur un
cluster de calcul. Elle sera accompagnée d’une procédure
d’installation, d’une documentation et d’une suite de tests
unitaires. Une image Docker sera également fournie pour en faciliter
le déploiement.


### Identification des rôles et organisation de l’ADT

Emmanuel Dupoux sera le responsable de l’ADT et coordonnera son
développement ainsi que sa valorisation. L’ingénieur recruté assurera
la responsabilité technique du développement logiciel et de
l’intégration des contributions ponctuelles (stagiaires, doctorants,
contributions via github). Des réunions d’avancement entre l’ingénieur
et M. Dupoux sont planifiées sur une base hebdomadaire. Certaines
phases de l’ADT vont impliquer la collaboration avec nos partenaires
(auteurs des modèles issus du challenge et intégrés à la toolbox),
parmi lesquels deux ont déjà donné leur accord de principe : Lucas
Ondel (Brno University of Technology) et Thomas Schatz (Université de
Maryland). L’ADT sera également supportée par une *Google Award*
obtenue récemment par l’équipe.

### Planification prévisionnelle

Le développement de Shennong se décompose en 6 jalons de 6 H/M chacun.

1. **T0 + 6 mois**: version 0.1 du logiciel prête et mise à
   disposition sur github. Cette version implémente le bloc
   *features*.

2. **T0 + 12 mois**: version 0.2 du logiciel prête et mise à
   disposition sur github. Cette version implémente le bloc *subword
   modelling*.

3. **T0 + 18 mois**: version 0.3 du logiciel prête et mise à
   disposition sur github. Cette version implémente le bloc *term
   discovery*.

4. **T0 + 24 mois**: rapport d’une étude comparative des performances
   des systèmes sur les données du challenge 2017.

5. **T0 + 30 mois**: préparation du prochain challenge : mise en place
   des bases de données de parole et implémentation des baselines et
   toplines.

6. **T0 + 36 mois**: préparation du prochain challenge : mise en ligne
   du site web, protocole de participation, automatisation des
   évaluations. Version 1.0 du logiciel et ouverture du *Zero Resource
   Challenge 2021*.


### Description et ordonnancement des tâches

#### 1. Bloc features (6 H/M, fin à T0+ 6 mois)

L’extraction de features à partir de signaux de paroles est la
première étape avant tout traitement par les blocs suivants. Ce
bloc intègre différents modèles dans une interface simple
harmonisant les entrée/sortie.

Pour ce bloc, la répartition des H/M est la suivante :

* 1 H/M sera consacré à la spécification de l’API, des formats de
  sortie et de la représentation en mémoire. Nous utiliserons le
  paquet h5features développé par l’équipe, il est également prévu
  d’inclure un convertisseur entre formats de données populaires
  (numpy/matlab/Kaldi/HTK/HDF5).

* 5 H/M seront chacun attribués à l’intégration dans le bloc d’un
  système existant de calcul de features. Les features requises sont :
  MFCC & filterbanks, PLP, BUT Bottleneck feature extractor du Speech
  Processing Group à Brno University of Technology, VTLN, 1 hot. Pour
  chacun des types de features à intégrer, lorsque c’est possible, une
  comparaison entre les différents systèmes existants et un choix
  justifié devra être fait.


#### 2. Bloc subword modeling (6 H/M, fin à T0+ 12 mois)

La tâche de subword modeling constitue la track 1 du Zero Resource
Speech Challenge. Ce second bloc prend en entrée les features
calculées au bloc précédent et implémente plusieurs algorithmes
représentatifs au sein d’une interface commune. Il intègre également
le système d’évaluation utilisé dans le challenge. Cette étape
consistera surtout à tester et documenter chaque système, à écrire un
wrapper pour avoir le même format d’entrée/sortie et générer une
méthode d’installation simple pour l’utilisateur. Ce bloc sera
considéré fini quand chacun des algorithmes envisagés sera intégré et
que les résultats du challenge seront reproductibles.

Pour ce bloc, la répartition des H/M est la suivante :

* 1 H/M sera consacré aux spécifications de l’API et des formats
  d’entrée/sortie.

* 1 H/M intégrera le logiciel ABXpy pour l’évaluation des modèles.

* 1 H/M sera consacré à l’intégration de briques permettant de
  construire des réseaux de neurones (intégration de dépendances
  telles que pytorch)

* 1 H/M sera consacré à l’intégration du projet ABnet3 (réseaux
  siamois)

* 1 H/M sera consacré à l’intégration d’une brique permettant de
  calculer des modèles DPGMM.

* 1 H/M sera consacré à l’intégration du système proposé par Chen et
  al. lors du challenge 2015 basé sur des DPGMM + VTLN.


#### 3. Bloc term discovery (6 H/M, fin à T0+ 18 mois)

La tâche de *term discovery* constitue la track 2 du *Zero Speech
Challenge*. D’un point de vue fonctionnel, ce troisième bloc est
similaire au deuxième et pourra s’appuyer sur les mêmes interfaces. De
la même manière, le système d’évaluation du challenge est également
intégré et nous fixons des impératifs de réplicabilité des résultats
du challenge. De plus, nous implémenterons des fonctionnalités
d’interaction entre les blocs *subword modeling* et *term discovery*
pour permettre un apprentissage croisé de ces 2 tâches. Plusieurs
interactions sont envisagées, séquentielles et parallèles, il s’agira
d’implémenter et de valider ces interactions.

Pour ce bloc, la répartition des H/M est la suivante :

* 1 H/M sera consacré aux spécifications de l’API et des formats
  d’entrée/sortie, à l’intégration du système d’évaluation tde et aux
  tests unitaires et d’intégrations

* 2 H/M seront consacrés au développement et à l’intégration de
  différents algorithmes pouvant être au coeur d’algorithmes de term
  discovery (KNN de FAISS, Segmental DTW …)

* 2 H/M seront consacrés à l’intégration des deux algorithmes étudiés
  par l’équipe de découverte des mots : le modèle d’Aren Jansen et la
  découverte des mots par KNN.

* 1 H/M sera consacré à la liaison subword modeling <-> term
  discovery. Toutes les combinaisons possibles doivent être envisagées
  et rendues possible pour les systèmes déjà intégrés et pour de
  futurs systèmes.

A l’issue de ce jalon, la toolbox sera fonctionnelle mais composée
d’un nombre minimal de modules. Nous envisageons, à cette étape, de
lancer un appel à contribution aux participants du challenge afin
qu’ils intègrent à Shennong leur propres modèles, en suivant un
canevas précis.


#### 4. Etude comparative sur 5 langages (6 H/M, fin à T0+ 24 mois)

Ce jalon peut être vu comme une preuve de concept de la toolbox. Il
s’agira de ré-implémenter une étude prospective[2] publiée par
l’équipe et de l’évaluer sur les données du *Zero Resource Challenge
2017* (5 langues). A l’instar de ce qui est fait pour Kaldi, cette
étape permettra aussi le développement de “recettes” qui seront
données en exemple dans la toolbox. Le pipeline proposé dans cette
étude exploite les 3 blocs de la toolbox comme ceci :

1. Bloc *features* : calcul des features à partir des fichiers wav du
   corpus,

2. Bloc *term discovery* : calcul de la matrice de distance du corpus
   avec lui-même. Détection de patterns diagonaux par filtrage et DTW,
   clustering des paires obtenues pour obtenir des proto-mots.

3. Bloc *subword modeling* : entraînement d’un réseau DNN siamois à
   partir des pairs de proto mots identiques ou différents. Recodage
   des features à partir des *embeddings* du DNN. Itération au point 2
   jusqu’à convergence.

Pour ce bloc, la répartition des H/M est la suivante :

* 2 H/M pour l’implémentation, le test et la validation des
  recettes étudiées.

* 2 H/M pour une étude systématique des recettes sur les datasets
  du challenge, comparaison aux résultats précédemment obtenus.

* 2 H/M pour la publication dans une version 0.4 de la toolbox des
  recettes créées avec une version fixe des différents paramètres,
  soumission d’un article dans un journal ou une conférence
  internationale.


#### 5. Edition 2021 du challenge : préparation

**Préparation des données, implémentation des baselines et toplines (6 H/M, fin à T0+ 30 mois).**

Ce jalon sera consacré aux travaux préparatoires en vue d’une nouvelle
édition du challenge Zero Resource, sur la base de cette nouvelle
toolbox. Il s’agira de préparer les données audio (choix des corpus,
nettoyage, pré-traitement, validation) et d’implémenter les baselines
et toplines qui serviront de référence aux participants du challenge.

Pour ce bloc, la répartition des H/M est la suivante :

* 2 H/M pour le choix des données, la spécification de leur format
  (ex: séparation “Relative”/”Outsider” proposée dans l’édition 2017).

* 2 H/M pour la préparation des corpora choisis.

* 2 H/M pour la spécification et l’implémentation des systèmes
  utilisés pour les baselines et toplines à partir de la toolbox
  Shennong, et le calcul des baselines et toplines sur les données
  préparées.


#### 6. Edition 2021 du challenge : déploiement

**Mise en place du site web et automatisation des tâches (6 H/M, fin
à T0+ 36 mois).**

Une fois les données du challenge préparées, un travail de mise en
place effective du challenge devra être réalisé pour le rendre
accessible aux participants. Le site web sera mis en place, son
hébergement se fera sur les serveurs de l’équipe à l’ENS. Les
spécifications de la plateforme utilisée pour le partage des données
du challenge seront faites en fonction du volume des données à
partager.

Un travail d’automatisation de l’évaluation sera aussi
effectué, de sorte à ce que les candidats puissent s’inscrire, nous
envoyer leur soumission, recevoir les résultats de l’évaluation et
voir les résultats s’afficher sur un leaderboard sans intervention
humaine de notre côté. En plus des articles publiés par les
participants, le challenge se conclura par la publication d’un article
synthétisant les contributions, résultats et perspectives comme ce fut
le cas lors des précédentes éditions.

Pour ce bloc, la répartition des H/M est la suivante :

* 2 H/M pour la création du site web sur lequel seront hébergés les
  conditions de participation et toutes les informations nécessaire à
  la participation.

* 2 H/M consacrés à l’automatisation des systèmes permettant de
  recevoir les participations des candidats, de les évaluer, et de
  publier un leaderboard.

* 2 H/M pour la mise en ligne du site web et la mise à disposition
  des corpora, systèmes d’évaluation et baselines.


### Anticipation des risques

Au niveau technique, l’intégration d’un nouveau modèle peut nécessiter
sa ré-écriture partielle (compilation/installation complexe, support
GPU, tests, documentation...). Par ailleurs cette toolbox agglomère un
certain nombre d’outils très différents et, bien qu’un soin
particulier sera porté aux dépendances logicielles et aux procédures
d’installation, il est probable qu’un problème de compatibilité
apparaisse sur une plateforme cible (MacOS, cluster, GPU…). Au niveau
scientifique il peut apparaître des problèmes de réplication de
résultats expérimentaux. Leur résolution peut nécessiter des
investigations importantes, comme l’équipe l’a déjà expérimenté
plusieurs fois.

Les risques évoqués ci-dessus sont susceptibles de ralentir le
développement du projet, mais ne compromettent pas son
aboutissement. A l’issue de l’ADT il est en effet impératif que
l’interface et au moins un algorithme/pipeline soit implémenté pour
chaque bloc. Mais, si un retard important intervient, il est tout à
fait envisageable de reporter l’intégration d’algorithmes
supplémentaires à une date post-ADT.

## Ressources

### Ressources mobilisées dans l’équipe

* Emmanuel Dupoux, chercheur, 10% du temps. Coordination,
  arbitrage, rédaction des livrables et articles.

* Rachid Riad, doctorant (2019-2022). Travail sur ABnet3 (bloc
  subword modeling) et primo-utilisateur.

* Xuan Nga Cao, ingénieure, ponctuel. Travail sur les données
  (corpus) lors des derniers jalons.

* Stagiaires (1 ou 2 stages de 6 mois par an). Les contributions des
  stagiaires peuvent être l’implémentation de modules supplémentaires
  dans la toolbox ou l’intégration de leur travail comme exemples
  d’utilisation. Les stagiaires seront par ailleurs les premiers
  utilisateurs et beta-testeurs de ce nouvel outil.

### Ressources humaines des services de l’INRIA

Aucune.


### Ressources demandées

Nous demandons pour cette ADT le recrutement d’un.e ingénieur.e
spécialiste pour une durée de 3 ans. Il/Elle travaillera à temps plein
au développement du projet au sein de l’équipe, localisée à l’ENS
Ulm. Etant donné la variété des tâches et l’hétérogénéité des outils
mis en oeuvre, nous recherchons un.e ingénieur.e spécialiste avec une
expérience en machine learning appliqué à la reconnaissance de la
parole. Il/Elle devra démontrer une solide connaissance de Python et
C/C++ et sera à l’aise avec les technologies de calcul parallèle
(clusters de calcul, accélération GPU) et le développement de projets
scientifiques dans un contexte open source.


### Ressources financières

Dans le but de faciliter le développement de l’ADT, nous souhaitons
faire une mise à jour de notre cluster de calcul afin de permettre
d’installer Docker et d’y déployer les tests de réplication prévus aux
différents jalons. En effet le cluster tourne sous CentOS-6.6 dont le
noyau en 2.6 est trop ancien pour Docker. Notre prestataire habituel
(Transtec) n’est plus en service et nous devrons nous adresser à
d’autres sociétés (par exemple NeoTeckno ou Alineos) pour obtenir des
devis détaillés. Nous estimons cette mise à jour à 30000 euros
maximum.


## Suivi et Évaluation

L’ADT verra son développement ouvert et hébergé sur github. Ainsi, la
mesure de l’activité pourra se faire à partir des statistiques
fournies par la plateforme (nombre de commits, nombre d’issues et de
contributeurs, taille de la base de code) ainsi que tirées du code
source (fonctionnalités et modèles implémentés), de la documentation
et des résultats répliqués.

Scientifiquement, l’impact du projet pourra se mesurer en terme de
publications : citations des articles associées à la toolbox et
nouvelles publications exploitant la toolbox ou l’un de ses
composants, qu’elles soient internes ou externes à l’équipe. L’impact
est également à envisager par les apports de contributeurs externes à
la toolbox. Ces contributions peuvent être l’intégration d’un modèle
par un tiers (un participant aux précédentes éditions du challenge par
exemple), la publication de scripts/pipelines utilisant la toolbox et,
dans une moindre mesure, les issues, forks et pull request effectués
sur github.

A la clôture de chaque jalon, la rédaction d’un bref livrable
synthétisant ces informations est envisageable. Enfin, au terme de
l’ADT, l’audience de la prochaine édition du challenge Zero Resource
et la création éventuelle d’une communauté d’utilisateurs autour de la
toolbox seront reportés.

Voici enfin une liste d’experts que nous pensons être en mesure
d’évaluer le bon déroulement de l’ADT :
* Benoît Sagot (INRIA, équipe Almanach)
* Irina Illina (INRIA, équipe Multispeech)
* Joseph Mariano (LIMSI)
* Laurent Besacier (Laboratoire d’Informatique de Grenoble)


## La sortie de l'ADT : positionnement après l’ADT

A l’issue de l’ADT, la toolbox aura été développée et publiée en open
source (jalons 1 à 3), testée et validée en interne (jalons 4 et 5)
puis en externe dans le cadre du challenge 2021 (jalons 6 et 7). Elle
offrira aux utilisateurs une baseline solide, fonctionnelle,
extensible et surtout reproductible. Ses outils d’évaluation
permettront de quantifier et comparer les performances des nouveaux
systèmes proposés par la communauté. Par ailleurs, ce nouveau
challenge amènera de nouvelles soumissions, donc de nouveaux systèmes
et potentiellement de nouvelles collaborations, ce qui permettra
d’inclure d’autant plus de système à la pointe de l’état de l’art à la
toolbox.

Les outils proposés en terme de découverte d’unités acoustiques et
lexicales permettront également l’ouverture vers de nouveaux thèmes de
recherche. Ainsi, un des buts de la toolbox Shennong est également
d’offrir à notre équipe, et à la communauté en général, une base
solide sur laquelle développer de nouveaux thèmes de recherche :

* Le *topic modeling*, c’est-à-dire la découverte du domaine d’une
  conversation ou de son champ lexical. Ce domaine pourra constituer à
  terme un 4ème bloc fonctionnel dans la toolbox, parallèle aux blocs
  de subword modeling et de term discovery.

* Le *crossmodal learning*, principalement audio-visuel et dans un cadre
  non supervisé ou faiblement supervisé. Les données visuelles peuvent
  être une image ou une brève vidéo illustrant la phrase (des
  approches similaires sont d'ores et déjà explorées pour le texte,
  notamment sur la base ImageNet), ou encore le mouvement des lèvres
  ou le visage du locuteur.

* L’*interactional learning*, qui est un domaine de recherche encore
  émergent, dans lequel un système artificiel apprend une tâche B (la
  reconnaissance de la parole) par l’intermédiaire d’une tâche A
  (jouer à un jeu), qui est vue alors comme un proxy.

Chacun de ces thèmes a fait l’objet d’études exploratoires dans notre
équipe (stages de fin d’étude ou thèses) et constituent une
orientation majeure de notre équipe à moyen terme (5-10 ans).

Finalement, la fin de l’ADT ne signifiera pas la fin du projet
Shennong, qui continuera d’être utilisé et maintenu par
l’équipe. Grâce au challenge et aux différentes collaborations créées
au fil du développement de la toolbox, nous souhaitons consolider une
communauté d’utilisateur et nous encouragerons les contributions
extérieures, à l’instar de Kaldi. Ainsi notre objectif à plus long
terme est de poursuivre le développement et l’enrichissement de cette
toolbox pour la maintenir à la pointe de l’état de l’art dans le
domaine de l’apprentissage non supervisé de la parole, en gardant en
tête ces principes que sont l’ouverture à la communauté et l’objectif
de réplicabilité des modèles et résultats.

### Notes

1. Shennong est le découvreur du thé dans la mythologie chinoise. Ce
   nom a été choisi par analogie à Kaldi, le découvreur du café et le
   nom d’une toolbox pour la reconnaissance de parole supervisée
   (http://kaldi-asr.org).

2. Thiollière, R., Dunbar, E., Synnaeve, G., Versteegh, M. & Dupoux,
   E. (2015). A Hybrid Dynamic Time Warping-Deep Neural Network
   Architecture for Unsupervised Acoustic Modeling. In
   INTERSPEECH-2015.
